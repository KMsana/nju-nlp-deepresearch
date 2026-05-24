# -*- coding: utf-8 -*-
"""
Multi-Agent Deep Research System — Planner / Executor / Assess / Synthesizer.

Combines the multi-agent framework with v3's proven pipeline:
  - Planner: decompose question → keyword queries
  - Screen agent: LLM picks ≤2 docs from search results (v3 Step B)
  - Executor: extract Facts + Dead Ends in NATURAL LANGUAGE (v3 Step D)
  - Assessor: constraint audit + gap analysis + next query (v3 Step E)
  - Synthesizer: final answer from structured evidence

Phase 1 enhancements:
  - ReRanker: BM25 top-15 → re-rank top-5 via term overlap
  - QueryAwareChunker: query-relevant chunks vs text[:6000]
  - EvidenceStore: structured {fact, docid, source_quote} tracking
  - Sliding-window query dedupe (window=2) vs permanent has_searched
  - Dead-loop guard: skip docids fetched ≥2 times
  - Real tool observations in trajectory
"""

import json
import re
from typing import Any, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════
# Memory
# ══════════════════════════════════════════════════════════════════════

class AgentMemory:
    """Structured memory across rounds."""

    def __init__(self):
        self.confirmed_facts: List[str] = []
        self.searched_queries: List[str] = []
        self.ruled_out: List[str] = []
        self.read_docids: List[str] = []
        self.pending_notes: List[str] = []
        self.evidence: List[Dict] = []
        self.fetched_docids: Dict[str, int] = {}

    def add_facts(self, facts: List[str]):
        for f in facts:
            if f and f not in self.confirmed_facts:
                self.confirmed_facts.append(f)

    def add_searched(self, query: str):
        q = query.strip()
        if q and q not in self.searched_queries:
            self.searched_queries.append(q)

    def was_recent(self, query: str, window: int = 2) -> bool:
        q = query.strip().lower()
        return any(q == r.lower() for r in self.searched_queries[-window:])

    def has_searched(self, query: str) -> bool:
        q = query.strip().lower()
        return any(q == s.lower() for s in self.searched_queries)

    def add_read(self, docid: str):
        if docid not in self.read_docids:
            self.read_docids.append(docid)

    def add_ruled_out(self, item: str):
        if item and item not in self.ruled_out:
            self.ruled_out.append(item)

    def add_note(self, note: str):
        if note:
            self.pending_notes.append(note)

    def add_evidence(self, fact: str, docid: str = "", source_quote: str = ""):
        key = fact.strip().lower()
        if any(e["fact"].strip().lower() == key for e in self.evidence):
            return False
        self.evidence.append({"fact": fact.strip(), "docid": docid,
                              "source_quote": source_quote})
        return True

    def facts_summary(self) -> str:
        if not self.confirmed_facts:
            return "(no confirmed facts yet)"
        return "\n".join(f"- {f}" for f in self.confirmed_facts)

    def searched_summary(self) -> str:
        if not self.searched_queries:
            return "(none)"
        return "\n".join(f"  [{i+1}] {q}"
                         for i, q in enumerate(self.searched_queries))

    def ruled_out_summary(self) -> str:
        if not self.ruled_out:
            return "(none)"
        return "\n".join(f"- {r}" for r in self.ruled_out)

    def evidence_summary(self, max_items: int = 10) -> str:
        if not self.evidence:
            return "(no structured evidence)"
        recent = self.evidence[-max_items:]
        return "\n".join(f"- {e['fact']}"
                         + (f" (doc: {e['docid']})" if e.get("docid") else "")
                         for e in recent)


# ══════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════

class ReRanker:
    @staticmethod
    def _normalize_scores(results: List[Dict]) -> List[float]:
        scores = [r["score"] for r in results]
        mn, mx = min(scores), max(scores)
        return [0.5] * len(scores) if mx == mn else [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _term_overlap(query: str, text: str) -> float:
        q_terms = set(query.lower().split())
        if not q_terms:
            return 0.0
        return len(q_terms & set(text.lower().split())) / len(q_terms)

    @classmethod
    def rerank(cls, query: str, results: List[Dict],
               top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        if not results:
            return []
        norm = cls._normalize_scores(results)
        for i, r in enumerate(results):
            overlap = cls._term_overlap(query, r.get("text", ""))
            r["rerank_score"] = alpha * norm[i] + (1 - alpha) * overlap
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]


class QueryAwareChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _score_chunk(query: str, chunk: str) -> float:
        q_terms = set(query.lower().split())
        if not q_terms:
            return 0.0
        return len(q_terms & set(chunk.lower().split()))

    @classmethod
    def select_relevant_chunks(cls, query: str, text: str, max_chunks: int = 5,
                               chunk_size: int = 1000, overlap: int = 200) -> str:
        chunks = cls.chunk_text(text, chunk_size, overlap)
        if len(chunks) <= max_chunks:
            return text
        scored = [(cls._score_chunk(query, c), i, c) for i, c in enumerate(chunks)]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:max_chunks]
        selected.sort(key=lambda x: x[1])
        return "\n".join(s[2] for s in selected)

    @classmethod
    def chunk_for_context(cls, query: str, text: str,
                          max_total_chars: int = 6000) -> str:
        max_chunks = max(1, max_total_chars // 1000)
        result = cls.select_relevant_chunks(query, text, max_chunks=max_chunks)
        return result[:max_total_chars] if len(result) > max_total_chars else result


# ══════════════════════════════════════════════════════════════════════
# Agent Prompts
# ══════════════════════════════════════════════════════════════════════

# ── Unified system prompt (v2/v3 style: one identity for all steps) ──
SYSTEM_PROMPT = (
    "You are a Deep Research Agent. Your task is to answer complex questions "
    "by searching a document corpus over multiple rounds. You maintain "
    "structured memory of confirmed facts across rounds."
)

PLANNER_PROMPT = """## Query Decomposition

Break the question below into 3-5 distinct search directions.

RULES:
- Each direction = 3-5 English keywords (NOT sentences, NOT questions)
- Cover different angles: different entities, time periods, events, concepts
- Use concrete, distinctive words that would appear VERBATIM in documents
- BM25 does pure keyword matching — rare distinctive words dominate

Example:
Question: A book published in the 1920s about Australian inland discoveries, by an author who married in the 1890s and wrote another book 1900-1910, with a barrel-shaped floating vessel on pages 332-339

Output:
- 1920s Australian inland exploration book
- author married 1890s publisher Methuen
- barrel-shaped floating vessel description
- botanist Allan Cunningham spear attack
- early Australian colonization 1906 book

Output one direction per line with "- " prefix. 3-5 keywords each."""

SCREEN_PROMPT = """## Document Screening

Review the search results above. Decide which documents are worth reading in full.
Select at most 2 documents — pick only those whose snippets contain the most specific, relevant content.

Output:
Relevant DocIDs: <comma-separated docids, or NONE>"""

EXECUTOR_PROMPT = """## Fact Extraction

Given the question and the full document texts above, classify your findings:

### Facts Found
List ONLY facts that match MULTIPLE constraints from the question — not just a single keyword.
- A name alone is NOT a fact unless the document also connects it to other question details (dates, places, relationships).
- If a document mentions a "librarian" but says nothing about Dakota, biography, or Europe visit — it is NOT a fact worth saving.
- Each fact must cite concrete details (names, dates, titles, events) confirmed by the document AND relevant to the specific constraints.
- If nothing matches multiple constraints, write "None."

### Dead Ends
Candidates found in the documents that are CONFIRMED NOT to match one or more question constraints. Only list if there is POSITIVE evidence a candidate is wrong (e.g., "Book X was published in 1935" when question requires 1920s). "Not mentioned" is NOT a dead end. If none, write "None."

Output:
Facts Found:
- fact 1

Dead Ends:
- candidate: why it violates which constraint"""

ASSESSOR_PROMPT = """## Progress Assessment

Check whether the confirmed facts satisfy all constraints, and decide next action.

RULES:
- List EVERY constraint from the question. Mark each as satisfied or no evidence.
- If ALL constraints are satisfied: READY_TO_ANSWER
- If any constraint lacks evidence: NEED_MORE
- If facts contradict a constraint, that candidate is ruled out. Switch direction.

BM25 does pure keyword matching — no semantics, no synonyms. Only exact word overlap counts. Rare distinctive words dominate.

For NEED_MORE: pick 3-6 most DISTINCTIVE keywords that would appear VERBATIM in the target document.

Output:
Constraint Audit:
- <constraint>: satisfied / no evidence

Status: NEED_MORE | READY_TO_ANSWER

-- If NEED_MORE:
Search Query: <keyword-query>"""

RETHINK_PROMPT = """## Rethink

You have been searching without results for multiple rounds. Look at the question, confirmed facts, query history, and ruled-out candidates. Identify a GENUINELY NEW search direction — a different entity, angle, or constraint.

Output:
New Direction: <what to pursue and why>
Search Query: <keyword-query>"""

SYNTHESIZER_PROMPT = """## Final Answer

All search rounds are complete. Based on all confirmed facts gathered through research, answer the question precisely.

CRITICAL RULES:
- Only use the facts listed above as evidence. DO NOT guess or hallucinate names.
- If ANY constraint from the question still has "no evidence", say: Unable to determine from available evidence.
- Only give a name/answer if ALL constraints are satisfied by confirmed facts.

Output:
Exact Answer: <the precise answer, or "Unable to determine from available evidence.">"""


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _chat(client, model, msgs, max_tok=2048):
    try:
        r = client.simple_chat(model=model, messages=msgs,
                               temperature=0.0, max_tokens=max_tok)
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _strip_think(text: str) -> str:
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    t = re.sub(r'<think>.*$', '', t, flags=re.DOTALL).strip()
    return t if t else text.strip()


def _parse_queries(text: str) -> List[str]:
    queries = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('-'):
            q = s[1:].strip().strip('"\'')
            words = q.split()
            if 2 <= len(words) <= 10:
                queries.append(q)
    return queries[:5]


def _parse_docids(text: str, results: List[Dict]) -> List[str]:
    m = re.search(r'Relevant DocIDs:\s*', text, re.IGNORECASE)
    if not m:
        return []
    rest = text[m.end():]
    stop_headers = ["Status:", "Next Query:", "Reasoning:", "Key Facts:",
                    "Explanation:", "Confidence:", "Thought:", "Action:",
                    "Facts Found:", "Dead Ends:", "Constraint Audit:",
                    "Exact Answer:", "New Direction:"]
    content = []
    for line in rest.split("\n"):
        s = line.strip()
        if not s:
            break
        if any(s.startswith(p) for p in stop_headers):
            break
        content.append(s)
    c = " ".join(content).strip()
    if not c or c.upper() == "NONE":
        return []

    docids = re.findall(r'\b(\d+)\b', c)
    result_docids = [d['docid'] for d in results]
    mapped, seen = [], set()
    for d in docids:
        if d in result_docids:
            actual = d
        elif d.isdigit() and 1 <= int(d) <= len(results):
            actual = results[int(d) - 1]['docid']
        else:
            continue
        if actual not in seen:
            seen.add(actual)
            mapped.append(actual)
    return mapped[:2]


def _parse_section(text: str, heading: str) -> List[str]:
    items, in_block = [], False
    pat = re.compile(re.escape(heading), re.I)
    for line in text.split("\n"):
        s = line.strip()
        if pat.search(s):
            in_block = True
            continue
        if in_block and s.startswith("-"):
            items.append(s[1:].strip())
        elif in_block and s and not s.startswith("-"):
            break
    return items


def _parse_facts(text: str) -> List[str]:
    for h in ["Facts Found", "facts found", "Key Facts", "key facts"]:
        f = _parse_section(text, h)
        if f:
            return [] if f in [["none"], ["None"], ["None."]] else f
    return []


def _parse_dead_ends(text: str) -> List[str]:
    d = _parse_section(text, "Dead Ends")
    return [] if d in [["none"], ["None"], ["None."]] else d


def _parse_status(text: str) -> str:
    m = re.search(r'^Status:\s*(NEED_MORE|READY_TO_ANSWER)\s*$',
                  text, re.I | re.M)
    return m.group(1).upper() if m else "NEED_MORE"


def _parse_search_query(text: str) -> str:
    for pat in [r'Search Query:\s*"?(.+?)"?\s*$',
                r'Search Query:\s*(.+?)\s*$']:
        m = re.search(pat, text, re.I | re.M)
        if m:
            return m.group(1).strip().strip('"\'')
    return ""


def _fallback_query(question: str, memory: AgentMemory) -> str:
    candidates = []
    candidates.extend(re.findall(r'"([^"]+)"', question))
    candidates.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                                 question))
    candidates.extend(re.findall(r'\b\d{3,4}\b', question))
    candidates.extend(re.findall(r'\b\d{1,2}(?:st|nd|rd|th)\b', question))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i+1]}")

    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'from', 'by', 'with',
            'and', 'or', 'but', 'not', 'this', 'that', 'these', 'those',
            'can', 'you', 'tell', 'find', 'what', 'when', 'where', 'who',
            'how', 'why', 'which', 'name', 'one', 'first', 'last', 'mid',
            'there', 'their', 'they', 'them', 'has', 'have', 'had', 'its',
            'also'}
    distinctive = []
    for c in candidates:
        cl = c.lower().strip()
        if cl not in stop and len(cl) > 2:
            distinctive.append(cl)

    out = []
    for t in distinctive:
        if not memory.was_recent(t) and t not in out:
            out.append(t)
        if len(out) >= 5:
            break
    return " ".join(out[:5]) if out else ""


def _format_results(results: List[Dict]) -> str:
    lines = []
    for i, doc in enumerate(results, 1):
        snippet = doc.get('snippet', doc.get('text', ''))[:500]
        lines.append(
            f"Result {i} | DocID: {doc['docid']} | Score: {doc['score']:.1f}\n"
            f"  {snippet}")
    return "\n\n".join(lines) if lines else "(no results)"


def _format_docs(docs: List[Dict], search_query: str = "",
                 max_chars: int = 6000) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", doc.get("error", ""))
        if search_query and text and "error" not in doc:
            text = QueryAwareChunker.chunk_for_context(
                search_query, text, max_total_chars=max_chars)
        elif len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        lines.append(f"--- Document {i} (docid={doc['docid']}) ---\n{text}")
    return "\n\n".join(lines) if lines else "(no documents)"


def _build_context(question: str, memory: AgentMemory,
                   extra: str = "") -> str:
    parts = [
        f"## Question\n{question}",
        f"\n## Confirmed Facts\n{memory.facts_summary()}",
    ]
    if memory.evidence:
        parts.append(f"\n## Structured Evidence\n{memory.evidence_summary()}")
    parts.extend([
        f"\n## Ruled Out\n{memory.ruled_out_summary()}",
        f"\n## Query History\n{memory.searched_summary()}",
    ])
    result = "\n".join(parts)
    if len(result) > 20000:
        facts = memory.confirmed_facts
        while len(result) > 20000 and len(facts) > 3:
            facts = facts[2:]
            parts[1] = f"\n## Confirmed Facts\n" + (
                "\n".join(f"- {f}" for f in facts) if facts else "(none)")
            result = "\n".join(parts)
    if extra:
        result += f"\n\n{extra}"
    return result


# ══════════════════════════════════════════════════════════════════════
# Multi-Agent functions
# ══════════════════════════════════════════════════════════════════════

def agent_plan(client, model, question: str,
               extra_context: str = "") -> List[str]:
    """Planner agent: decompose question into keyword queries."""
    user = f"## Question\n{question}\n\n{PLANNER_PROMPT}"
    if extra_context:
        user = (f"## Question\n{question}\n\n"
                f"## What We Already Know\n{extra_context}\n\n"
                f"{PLANNER_PROMPT}\nFocus on what is still MISSING.")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    raw = _strip_think(_chat(client, model, msgs, max_tok=512))
    return _parse_queries(raw)


def agent_retrieve(registry: Dict, query: str, memory: AgentMemory,
                   searcher=None) -> List[Dict]:
    memory.add_searched(query)
    search_fn = registry["search"]
    if searcher:
        try: raw = searcher.search(query, k=15)
        except Exception: raw = []
    else:
        try: raw = search_fn(query)
        except Exception: raw = []
    if not raw:
        return []
    return ReRanker.rerank(query, raw, top_k=5) if len(raw) > 5 else raw


def agent_screen(client, model, question: str, results: List[Dict],
                 memory: AgentMemory) -> str:
    ctx = _build_context(question, memory)
    prompt = (f"{ctx}\n\n## Search Results\n{_format_results(results)}\n\n"
              f"{SCREEN_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=512)


def agent_fetch(registry: Dict, docids: List[str],
                memory: AgentMemory) -> List[Dict]:
    get_doc_fn = registry["get_document"]
    docs = []
    for did in docids[:3]:
        fc = memory.fetched_docids.get(did, 0)
        if fc >= 2: continue
        try: doc = get_doc_fn(did)
        except Exception: doc = None
        if doc is None or (isinstance(doc, dict) and "error" in doc):
            docs.append({"docid": did, "error": "not found"})
        else:
            memory.add_read(did)
            memory.fetched_docids[did] = fc + 1
            docs.append({"docid": did, "text": doc.get("text", ""),
                         "url": doc.get("url", "")})
    return docs


def agent_execute(client, model, question: str, docs: List[Dict],
                  memory: AgentMemory, search_query: str = "") -> str:
    if not docs: return ""
    ctx = _build_context(question, memory)
    prompt = (f"{ctx}\n\n"
              f"## Full Documents\n{_format_docs(docs, search_query)}\n\n"
              f"{EXECUTOR_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def agent_assess(client, model, question: str, memory: AgentMemory,
                 current_query: str) -> str:
    extra = (f"Most recent query: \"{current_query}\"\n"
             f"Rounds searched: {len(memory.searched_queries)}")
    ctx = _build_context(question, memory, extra=extra)
    prompt = f"{ctx}\n\n{ASSESSOR_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=1024)


def agent_rethink(client, model, question: str,
                  memory: AgentMemory) -> str:
    ctx = _build_context(question, memory)
    prompt = f"{ctx}\n\n{RETHINK_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    raw = _chat(client, model, msgs, max_tok=512)
    return _parse_search_query(raw)


def agent_synthesize(client, model, question: str,
                     memory: AgentMemory) -> str:
    ctx = _build_context(question, memory)
    prompt = f"{ctx}\n\n{SYNTHESIZER_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=512)


# ══════════════════════════════════════════════════════════════════════
# Trajectory builder
# ══════════════════════════════════════════════════════════════════════

def _build_trajectory(question: str, memory: AgentMemory,
                      round_records: List[Dict],
                      final_answer: str) -> List[Dict]:
    messages: List[Dict] = [
        {"role": "system",
         "content": "Multi-agent deep research system — Planner/Screen/Executor/Assess/Synthesizer."},
        {"role": "user", "content": question},
    ]
    call_id = 0
    for rec in round_records:
        # Search
        for q in rec.get("queries", []):
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                                "function": {"name": "search",
                                             "arguments": json.dumps(
                                                 {"query": q},
                                                 ensure_ascii=False)}}]})
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps([
                    {"docid": d["docid"], "score": d.get("score", 0),
                     "snippet": (d.get("text", "") or "")[:300]}
                    for d in rec.get("results", [])
                ], ensure_ascii=False)})
        # Screen
        messages.append({"role": "assistant", "content": rec.get("screen", "")})
        # Get document
        fetched = rec.get("fetched", [])
        if fetched:
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                                "function": {"name": "get_document",
                                             "arguments": json.dumps(
                                                 {"docid": d["docid"]},
                                                 ensure_ascii=False)}}
                               for d in fetched]})
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps([
                    {"docid": d["docid"],
                     "text_preview": (d.get("text", "") or "")[:500],
                     "url": d.get("url", "")}
                    for d in fetched], ensure_ascii=False)})
        # Executor output
        messages.append({"role": "assistant", "content": rec.get("extract", "")})
        # Assessor output
        messages.append({"role": "assistant", "content": rec.get("assess", "")})

    messages.append({"role": "assistant", "content": final_answer})
    return messages


# ══════════════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════════════

def run_agent_loop(
    client, model, query: str,
    tools: List[Dict], registry: Dict[str, Any],
    max_turns: int = 5, max_history_msgs: int = 6,
) -> Tuple[str, List[Dict]]:
    """Multi-agent deep research — Planner/Screen/Executor/Assess/Synthesizer."""

    memory = AgentMemory()
    round_records: List[Dict] = []
    empty_rounds = 0
    suggested_query = ""
    searcher = registry.get("_searcher")

    # Phase 0: Planner decomposes question → keyword query pool
    angle_pool = agent_plan(client, model, query)
    angle_idx = 0

    for round_num in range(1, max_turns + 1):
        # ── Determine query ──
        if round_num == 1:
            # R1: use first Planner output, or full question as fallback
            if angle_pool:
                current_query = angle_pool[0]
                angle_idx = 1
            else:
                current_query = query
        else:
            current_query = suggested_query

        if not current_query or not current_query.strip():
            fb = _fallback_query(query, memory)
            if fb and not memory.was_recent(fb):
                current_query = fb
            else:
                break

        if memory.was_recent(current_query):
            # Try next Planner angle
            if angle_idx < len(angle_pool):
                current_query = angle_pool[angle_idx]
                angle_idx += 1
            else:
                fb = _fallback_query(query, memory)
                if fb and not memory.was_recent(fb):
                    current_query = fb
                else:
                    break

        # Stuck: try rethink
        if round_num > 1 and empty_rounds >= 2:
            rethink_q = agent_rethink(client, model, query, memory)
            if rethink_q and not memory.was_recent(rethink_q):
                current_query = rethink_q
                memory.add_note("Forced rethink — new direction")

        # Empty rounds → try next Planner angle
        if round_num > 1 and empty_rounds >= 1 and not memory.was_recent(current_query):
            if suggested_query and not memory.was_recent(suggested_query):
                current_query = suggested_query
            elif angle_idx < len(angle_pool):
                current_query = angle_pool[angle_idx]
                angle_idx += 1
                memory.add_note(f"Switching angle {angle_idx}/{len(angle_pool)}")

        rec: Dict = {"queries": [current_query]}

        # ── Retrieve + Rerank ──
        results = agent_retrieve(registry, current_query, memory, searcher)
        rec["results"] = results
        if not results:
            empty_rounds += 1
            continue

        # ── Screen: LLM picks docs ──
        screen_raw = agent_screen(client, model, query, results, memory)
        screen_raw = _strip_think(screen_raw)
        rec["screen"] = screen_raw
        docids = _parse_docids(screen_raw, results)

        # ── Fetch (QueryAwareChunker + dead-loop guard) ──
        docs = agent_fetch(registry, docids, memory) if docids else []
        rec["fetched"] = docs

        # ── Execute: extract Facts + Dead Ends ──
        extract_raw = agent_execute(client, model, query, docs, memory,
                                    current_query) if docs else ""
        extract_raw = _strip_think(extract_raw)
        rec["extract"] = extract_raw
        new_facts = _parse_facts(extract_raw)
        dead_ends = _parse_dead_ends(extract_raw)
        memory.add_facts(new_facts)
        for d in dead_ends:
            memory.add_ruled_out(d)
        for f in new_facts:
            memory.add_evidence(f)

        if new_facts:
            empty_rounds = 0
        else:
            empty_rounds += 1

        if empty_rounds >= 1 and memory.confirmed_facts:
            memory.add_note("WARNING: no new facts — try different angle")
        if dead_ends:
            memory.add_note(
                f"Dead ends: {', '.join(d for d in dead_ends[:3])}")

        # ── Assess: constraint audit ──
        assess_raw = agent_assess(client, model, query, memory, current_query)
        assess_raw = _strip_think(assess_raw)
        rec["assess"] = assess_raw

        round_records.append(rec)
        status = _parse_status(assess_raw)

        if status == "READY_TO_ANSWER":
            break

        # ── Assessor → Planner feedback loop ──
        # Parse what's still missing, feed to Planner for targeted re-decomposition
        missing_info = _parse_section(assess_raw, "Missing Information")
        next_q = _parse_search_query(assess_raw)

        if missing_info and round_num >= 2:
            # Feed missing info back to Planner → new targeted queries
            ctx = "MISSING: " + "; ".join(missing_info[:5])
            new_angles = agent_plan(client, model, query, extra_context=ctx)
            if new_angles:
                # Prepend new angles to pool (skip recent ones)
                fresh = [q for q in new_angles if not memory.was_recent(q)]
                if fresh:
                    angle_pool = fresh + angle_pool
                    angle_idx = 0

        if next_q and not memory.was_recent(next_q):
            suggested_query = next_q
        elif angle_idx < len(angle_pool):
            suggested_query = angle_pool[angle_idx]
            angle_idx += 1
        else:
            fb = _fallback_query(query, memory)
            if fb and not memory.was_recent(fb):
                suggested_query = fb
            else:
                break

    # ── Synthesize ──
    answer_raw = agent_synthesize(client, model, query, memory)
    final_answer = _strip_think(answer_raw)
    m = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', final_answer, re.IGNORECASE)
    if m:
        final_answer = m.group(1).strip()

    trajectory = _build_trajectory(query, memory, round_records, final_answer)
    return final_answer, trajectory
