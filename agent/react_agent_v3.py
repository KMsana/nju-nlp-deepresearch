# -*- coding: utf-8 -*-
"""
Deep Research Agent — 5-step pipeline per round with structured memory.
v3: adds pre-search question decomposition into multiple keyword angles.

Pipeline per round:
  Step A: code-forced search(query)
  Step B: model screens results → picks up to 3 docids to read
  Step C: code-forced get_document for each selected docid
  Step D: model extracts key facts + dead ends from full documents
  Step E: model audits constraints → NEED_MORE / READY_TO_ANSWER

Round 1 uses the full question as the search query.
When stuck, cycles through pre-decomposed keyword angles before rethink.
Structured memory (facts, searched queries, ruled-out) maintained across rounds.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ── AgentMemory ──────────────────────────────────────────────────────

class AgentMemory:
    """Structured memory persisted across search rounds."""

    def __init__(self) -> None:
        self.confirmed_facts: List[str] = []
        self.searched_queries: List[str] = []
        self.ruled_out: List[str] = []       # candidates confirmed NOT to match
        self.read_docids: List[str] = []      # docids already fetched
        self.pending_notes: List[str] = []    # warnings / hints for next round

    def add_facts(self, facts: List[str]) -> None:
        for f in facts:
            if f and f not in self.confirmed_facts:
                self.confirmed_facts.append(f)

    def add_searched(self, query: str) -> None:
        q = query.strip()
        if q and q not in self.searched_queries:
            self.searched_queries.append(q)

    def has_searched(self, query: str) -> bool:
        q = query.strip().lower()
        return any(q == s.lower() for s in self.searched_queries)

    def add_read(self, docid: str) -> None:
        if docid not in self.read_docids:
            self.read_docids.append(docid)

    def add_ruled_out(self, item: str) -> None:
        if item and item not in self.ruled_out:
            self.ruled_out.append(item)

    def add_note(self, note: str) -> None:
        if note:
            self.pending_notes.append(note)

    def facts_summary(self) -> str:
        if not self.confirmed_facts:
            return "(no confirmed facts yet)"
        return "\n".join(f"- {f}" for f in self.confirmed_facts)

    def searched_summary(self) -> str:
        if not self.searched_queries:
            return "(none)"
        return "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(self.searched_queries))

    def ruled_out_summary(self) -> str:
        if not self.ruled_out:
            return "(none)"
        return "\n".join(f"- {r}" for r in self.ruled_out)


# ── Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Deep Research Agent. Your task is to answer complex questions "
    "by searching a document corpus over multiple rounds. You maintain "
    "structured memory of confirmed facts across rounds."
)

# ── v3 NEW: Question decomposition prompt ─────────────────────────
DECOMPOSE_PROMPT = """## Query Decomposition

Break the question below into 3-5 distinct search directions.

RULES:
- Each direction = 3-5 English keywords (NOT sentences, NOT questions)
- Cover different angles: different entities, time periods, events, concepts
- Use concrete, distinctive words that would appear VERBATIM in documents
- BM25 does pure keyword matching — rare distinctive words dominate

Example:
Question: A book published in the 1920s about Australian inland discoveries,
         by an author who married in the 1890s and wrote another book 1900-1910,
         with a barrel-shaped floating vessel on pages 332-339

Output:
- 1920s Australian inland exploration book
- author married 1890s publisher Methuen
- barrel-shaped floating vessel description
- botanist Allan Cunningham spear attack
- early Australian colonization 1906 book

Output one direction per line with "- " prefix. 3-5 keywords each."""

DOC_SCREEN_PROMPT = """## Document Screening

Review the search results above. Decide which documents are worth reading in full.
Select at most 2 documents — pick only those whose snippets contain the most specific, relevant content.

Output:
Relevant DocIDs: <comma-separated docids, or NONE>"""

FACT_EXTRACT_PROMPT = """## Fact Extraction

Given the question and the full document texts above, classify your findings:

### Facts Found
Specific, verifiable facts FROM the documents that are RELATED to the question. Each fact must cite concrete details (names, dates, titles, events) that appear in both the document and the question. If nothing relevant, write "None."

### Dead Ends
Candidates found in the documents that are CONFIRMED NOT to match one or more question constraints. Only list if there is POSITIVE evidence a candidate is wrong (e.g., "Book X was published in 1935" when question requires 1920s). "Not mentioned" is NOT a dead end. If none, write "None."

Output:
Facts Found:
- fact 1
- fact 2

Dead Ends:
- candidate: why it violates which constraint"""

PROGRESS_PROMPT = """## Progress Assessment

Check whether the confirmed facts satisfy all constraints, and decide next action.

RULES:
- List EVERY constraint from the question. Mark each as satisfied or no evidence.
- If ALL constraints are satisfied: READY_TO_ANSWER
- If any constraint lacks evidence: NEED_MORE
- If facts contradict a constraint, that candidate is ruled out. Switch direction.

BM25 does pure keyword matching — no semantics, no synonyms. Only exact word overlap counts. Rare distinctive words dominate.

For NEED_MORE: pick 3-5 most DISTINCTIVE keywords that would appear VERBATIM in the target document.

Output:
Constraint Audit:
- <constraint>: satisfied / no evidence

Status: NEED_MORE | READY_TO_ANSWER

-- If NEED_MORE:
Search Query: <3-5 keywords>"""

FINAL_ANSWER_PROMPT = """## Final Answer

All search rounds are complete. Based on all confirmed facts gathered through research, answer the question precisely.
Only use the facts listed above as evidence. If evidence is insufficient, say so honestly.

Output:
Exact Answer: <the precise answer>"""

RETHINK_PROMPT = """## Rethink

You have been searching without results for multiple rounds. Look at the question, confirmed facts, query history, and ruled-out candidates. Identify a GENUINELY NEW search direction — a different entity, angle, or constraint.

Output:
New Direction: <what to pursue and why>
Search Query: <3-5 keywords>"""


# ── Helpers ──────────────────────────────────────────────────────────

def _chat(client: Any, model: str, messages: List[Dict[str, Any]],
          max_tokens: int = 2048) -> str:
    try:
        resp = client.simple_chat(model=model, messages=messages,
                                   temperature=0.0, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _strip_think(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
    return cleaned  # don't fall back to raw <think> text


# ── v3 NEW: Parse decomposition output ─────────────────────────────
def _parse_decompose_queries(text: str) -> List[str]:
    """Parse '- keyword1 keyword2 ...' lines from decomposition output."""
    queries = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('-'):
            q = s[1:].strip().strip('"\'')
            words = q.split()
            if 2 <= len(words) <= 10:
                queries.append(q)
    return queries[:5]


def _step_decompose(client: Any, model: str, question: str) -> List[str]:
    """Phase 0: 1 LLM call to decompose question into 3-5 keyword queries."""
    msgs = [
        {"role": "system", "content": "You decompose complex questions into keyword search queries."},
        {"role": "user", "content": f"Decompose this question into 3-5 keyword search directions (3-5 words each):\n\n{question}\n\n{DECOMPOSE_PROMPT}"}
    ]
    raw = _strip_think(_chat(client, model, msgs, max_tokens=512))
    queries = _parse_decompose_queries(raw)
    return queries if queries else []


def _parse_docids(text: str, results: List[Dict[str, Any]]) -> List[str]:
    """Parse 'Relevant DocIDs: 41740, 18896' or 'Relevant DocIDs: NONE'."""
    m = re.search(r'Relevant DocIDs:\s*', text, re.IGNORECASE)
    if not m:
        return []
    rest = text[m.end():]
    # Stop at next section header
    stop = ["Status:", "Next Query:", "Reasoning:", "Key Facts:",
            "Explanation:", "Confidence:", "Thought:", "Action:",
            "Facts Found:", "Dead Ends:", "Constraint Audit:", "Exact Answer:"]
    content = []
    for line in rest.split("\n"):
        s = line.strip()
        if not s:
            break
        if any(s.startswith(p) for p in stop):
            break
        content.append(s)
    content_str = " ".join(content).strip()
    if not content_str or content_str.upper() == "NONE":
        return []

    docids = re.findall(r'\b(\d+)\b', content_str)
    result_docids = [d['docid'] for d in results]
    mapped = []
    seen = set()
    for d in docids:
        if d in result_docids:
            actual = d
        elif d.isdigit():
            idx = int(d)
            if 1 <= idx <= len(results):
                actual = results[idx - 1]['docid']
            else:
                continue
        else:
            continue
        if actual not in seen:
            seen.add(actual)
            mapped.append(actual)
    return mapped[:2]


def _parse_section(text: str, heading: str) -> List[str]:
    """Parse bullet-list section from model output."""
    items = []
    in_block = False
    pattern = re.compile(re.escape(heading), re.IGNORECASE)
    for line in text.split("\n"):
        s = line.strip()
        if pattern.search(s):
            in_block = True
            continue
        if in_block and s.startswith("-"):
            items.append(s[1:].strip())
        elif in_block and s and not s.startswith("-"):
            break
    return items


def _parse_facts(text: str) -> List[str]:
    for h in ["Facts Found", "facts found", "Key Facts", "key facts"]:
        facts = _parse_section(text, h)
        if facts:
            return [] if facts in [["none"], ["None"], ["None."]] else facts
    return []


def _parse_dead_ends(text: str) -> List[str]:
    items = _parse_section(text, "Dead Ends")
    return [] if items in [["none"], ["None"], ["None."]] else items


def _parse_status(text: str) -> str:
    m = re.search(r'^Status:\s*(NEED_MORE|READY_TO_ANSWER)\s*$',
                  text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).upper() if m else "NEED_MORE"


def _parse_search_query(text: str) -> str:
    for pat in [r'Search Query:\s*"?(.+?)"?\s*$', r'Search Query:\s*(.+?)\s*$']:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().strip('"\'')
    return ""


def _fallback_query(question: str, memory: AgentMemory) -> str:
    """Extract distinctive terms from question as BM25 keywords."""
    # Proper nouns, capitalized phrases, numbers
    candidates = []
    candidates.extend(re.findall(r'"([^"]+)"', question))
    candidates.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question))
    candidates.extend(re.findall(r'\b\d{3,4}\b', question))
    candidates.extend(re.findall(r'\b\d{1,2}(?:st|nd|rd|th)\b', question))
    # Also extract noun phrases (simple: 2+ word sequences with no stopwords)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i+1]}")

    stop = {'the','a','an','is','was','are','were','be','been','in','on','at',
            'to','for','of','from','by','with','and','or','but','not','this',
            'that','these','those','can','you','tell','find','what','when',
            'where','who','how','why','which','name','one','first','last','mid',
            'there','their','they','them','has','have','had','its','also'}
    distinctive = []
    for c in candidates:
        cl = c.lower().strip()
        if cl not in stop and len(cl) > 2:
            distinctive.append(cl)

    unsearched = []
    for t in distinctive:
        if not any(t in q.lower() for q in memory.searched_queries):
            if t not in unsearched:
                unsearched.append(t)
            if len(unsearched) >= 5:
                break
    return " ".join(unsearched[:5]) if unsearched else ""


def _format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results with snippet for screening."""
    lines = []
    for i, doc in enumerate(results, 1):
        snippet = doc.get('snippet', doc.get('text', ''))[:500]
        lines.append(
            f"Result {i} | DocID: {doc['docid']} | Score: {doc['score']:.1f}\n"
            f"  {snippet}"
        )
    return "\n\n".join(lines) if lines else "(no results)"


def _format_docs(docs: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", doc.get("error", ""))
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        lines.append(f"--- Document {i} (docid={doc['docid']}) ---\n{text}")
    return "\n\n".join(lines) if lines else "(no documents)"


def _build_context(question: str, memory: AgentMemory,
                   extra: str = "") -> str:
    parts = [
        f"## Question\n{question}",
        f"\n## Confirmed Facts\n{memory.facts_summary()}",
        f"\n## Ruled Out\n{memory.ruled_out_summary()}",
        f"\n## Query History\n{memory.searched_summary()}",
    ]
    # Truncate context if too long (>30000 chars): trim older facts first
    result = "\n".join(parts)
    if len(result) > 20000:
        # Trim facts to last N
        facts = memory.confirmed_facts
        while len(result) > 20000 and len(facts) > 3:
            facts = facts[2:]
            parts[1] = f"\n## Confirmed Facts\n" + ("\n".join(f"- {f}" for f in facts) if facts else "(none)")
            result = "\n".join(parts)
    if extra:
        result += f"\n\n{extra}"
    return result


# ── Pipeline steps ───────────────────────────────────────────────────

def _step_search(registry: Dict[str, Any], query: str,
                 memory: AgentMemory) -> List[Dict[str, Any]]:
    """Step A: code-forced search."""
    memory.add_searched(query)
    search_fn = registry["search"]
    try:
        return search_fn(query)
    except Exception:
        return []


def _step_screen(client: Any, model: str, question: str,
                 results: List[Dict[str, Any]],
                 memory: AgentMemory) -> str:
    """Step B: model screens results, returns raw output."""
    ctx = _build_context(question, memory)
    prompt = (f"{ctx}\n\n"
              f"## Search Results\n{_format_results(results)}\n\n"
              f"{DOC_SCREEN_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tokens=2048)


def _step_fetch(registry: Dict[str, Any], docids: List[str],
                memory: AgentMemory) -> List[Dict[str, Any]]:
    """Step C: code-forced get_document."""
    get_doc_fn = registry["get_document"]
    docs = []
    for docid in docids[:3]:
        try:
            doc = get_doc_fn(docid)
        except Exception:
            doc = None
        if doc is None or (isinstance(doc, dict) and "error" in doc):
            docs.append({"docid": docid, "error": "not found"})
        else:
            memory.add_read(docid)
            docs.append({
                "docid": docid,
                "text": doc.get("text", ""),
                "url": doc.get("url", ""),
            })
    return docs


def _step_extract(client: Any, model: str, question: str,
                  docs: List[Dict[str, Any]],
                  memory: AgentMemory) -> str:
    """Step D: model extracts facts + dead ends, returns raw output."""
    if not docs:
        return ""
    ctx = _build_context(question, memory)
    prompt = (f"{ctx}\n\n"
              f"## Full Documents\n{_format_docs(docs)}\n\n"
              f"{FACT_EXTRACT_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tokens=2048)


def _step_assess(client: Any, model: str, question: str,
                 memory: AgentMemory, current_query: str) -> str:
    """Step E: model assesses progress, returns raw output."""
    extra = (f"Most recent query: \"{current_query}\"\n"
             f"Rounds searched: {len(memory.searched_queries)}")
    ctx = _build_context(question, memory, extra=extra)
    prompt = f"{ctx}\n\n{PROGRESS_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tokens=2048)


def _step_final_answer(client: Any, model: str, question: str,
                       memory: AgentMemory) -> str:
    """Generate final answer from accumulated facts."""
    ctx = _build_context(question, memory)
    prompt = f"{ctx}\n\n{FINAL_ANSWER_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tokens=1024)


def _step_rethink(client: Any, model: str, question: str,
                  memory: AgentMemory) -> str:
    """Force model to find a new search direction."""
    ctx = _build_context(question, memory)
    prompt = f"{ctx}\n\n{RETHINK_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    raw = _chat(client, model, msgs, max_tokens=1024)
    return _parse_search_query(raw)


# ── Trajectory builder ───────────────────────────────────────────────

def _build_trajectory(question: str, memory: AgentMemory,
                      round_records: List[Dict[str, Any]],
                      final_answer: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    call_id = 0
    for rec in round_records:
        # Step A: search
        call_id += 1
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                "function": {"name": "search",
                    "arguments": json.dumps({"query": rec["query"]}, ensure_ascii=False)}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": f"call_{call_id}",
            "content": json.dumps([{"docid": d["docid"], "score": d["score"]}
                                   for d in rec.get("results", [])], ensure_ascii=False),
        })
        # Step B: screen
        messages.append({"role": "assistant", "content": rec.get("screen", "")})
        # Step C: get_document
        fetched = rec.get("fetched")
        if fetched:
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                    "function": {"name": "get_document",
                        "arguments": json.dumps({"docid": d["docid"]}, ensure_ascii=False)}} for d in fetched],
            })
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps([{"docid": d["docid"]} for d in fetched],
                                      ensure_ascii=False),
            })
        # Step D: facts
        messages.append({"role": "assistant", "content": rec.get("extract", "")})
        # Step E: assess
        messages.append({"role": "assistant", "content": rec.get("assess", "")})

    messages.append({"role": "assistant", "content": final_answer})
    return messages


# ── Main entry ───────────────────────────────────────────────────────

def run_agent_loop(
    client: Any, model: str, query: str,
    tools: List[Dict[str, Any]], registry: Dict[str, Any],
    max_turns: int = 5, max_history_msgs: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    """5-step deep research pipeline with structured memory.
    v3: pre-decomposes the question into keyword angles, cycles through them when stuck."""

    memory = AgentMemory()
    round_records: List[Dict[str, Any]] = []
    empty_rounds = 0
    suggested_query = ""  # from previous Step E

    # ── v3 NEW: Phase 0 - decompose question into keyword angles ──
    angle_pool = _step_decompose(client, model, query)
    angle_idx = 0  # which pre-decomposed angle to try next

    for round_num in range(1, max_turns + 1):
        # ── Determine query ──
        if round_num == 1:
            current_query = query  # Round 1: full question (v2 default)
        else:
            current_query = suggested_query

        if not current_query or not current_query.strip():
            fb = _fallback_query(query, memory)
            if fb and not memory.has_searched(fb):
                current_query = fb
            else:
                break

        if memory.has_searched(current_query):
            fb = _fallback_query(query, memory)
            if fb and not memory.has_searched(fb):
                current_query = fb
            else:
                break

        # ── v3 NEW: no facts → immediately switch to next pre-decomposed angle ──
        if round_num > 1 and empty_rounds >= 1:
            # If model generated a query AND it differs from what we just searched, try it first
            if suggested_query and not memory.has_searched(suggested_query):
                current_query = suggested_query
            elif angle_idx < len(angle_pool):
                next_angle = angle_pool[angle_idx]
                angle_idx += 1
                if not memory.has_searched(next_angle):
                    current_query = next_angle
                    memory.add_note(f"Switching angle {angle_idx}/{len(angle_pool)}: {next_angle}")
            elif not current_query or memory.has_searched(current_query):
                fb = _fallback_query(query, memory)
                if fb and not memory.has_searched(fb):
                    current_query = fb

        # Rethink after 2+ empty rounds (v2 original logic — still here)
        if round_num > 1 and empty_rounds >= 2:
            rethink_q = _step_rethink(client, model, query, memory)
            if rethink_q and not memory.has_searched(rethink_q):
                current_query = rethink_q
                memory.add_note("Forced rethink — new direction")

        rec: Dict[str, Any] = {"query": current_query}

        # ── Step A: Search ──
        results = _step_search(registry, current_query, memory)
        rec["results"] = results
        if not results:
            empty_rounds += 1
            continue

        # ── Step B: Screen ──
        screen_raw = _step_screen(client, model, query, results, memory)
        screen_raw = _strip_think(screen_raw)
        rec["screen"] = screen_raw
        docids = _parse_docids(screen_raw, results)

        # ── Step C: Fetch ──
        docs = _step_fetch(registry, docids, memory) if docids else []
        rec["fetched"] = docs

        # ── Step D: Extract facts ──
        extract_raw = _step_extract(client, model, query, docs, memory) if docs else ""
        extract_raw = _strip_think(extract_raw)
        rec["extract"] = extract_raw
        new_facts = _parse_facts(extract_raw)
        dead_ends = _parse_dead_ends(extract_raw)
        memory.add_facts(new_facts)
        for d in dead_ends:
            memory.add_ruled_out(d)

        if new_facts:
            empty_rounds = 0
        else:
            empty_rounds += 1

        if empty_rounds >= 1 and memory.confirmed_facts:
            memory.add_note("WARNING: no new facts last round — try a different search angle")
        if dead_ends:
            memory.add_note(f"Dead ends this round: {', '.join(d for d in dead_ends[:3])}")

        # ── Step E: Assess ──
        assess_raw = _step_assess(client, model, query, memory, current_query)
        assess_raw = _strip_think(assess_raw)
        rec["assess"] = assess_raw

        round_records.append(rec)
        status = _parse_status(assess_raw)

        if status == "READY_TO_ANSWER":
            break

        # Prepare next query
        next_q = _parse_search_query(assess_raw)
        if not next_q:
            fb = _fallback_query(query, memory)
            if fb and not memory.has_searched(fb):
                next_q = fb
            else:
                break
        suggested_query = next_q

    # ── Final answer ──
    answer_raw = _step_final_answer(client, model, query, memory)
    final_answer = _strip_think(answer_raw)

    # Extract clean answer from "Exact Answer:" format
    m = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', final_answer, re.IGNORECASE)
    if m:
        final_answer = m.group(1).strip()

    # If still empty (all <think>), extract last meaningful content from records
    if not final_answer.strip():
        for rec in reversed(round_records):
            for key in ("assess", "extract"):
                txt = rec.get(key, "")
                m = re.search(r'Exact Answer:\s*(.+)', txt, re.I)
                if not m:
                    # Try to get any factual statement
                    for line in txt.split("\n"):
                        if line.strip().startswith("- ") and len(line) > 10:
                            final_answer = line.strip()
                            break
                if final_answer.strip():
                    break
            if final_answer.strip():
                break
        if not final_answer.strip():
            final_answer = "Unable to determine answer from available evidence."

    trajectory = _build_trajectory(query, memory, round_records, final_answer)
    return final_answer, trajectory
