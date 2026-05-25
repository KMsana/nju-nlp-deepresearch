# -*- coding: utf-8 -*-
"""
Deep Research Agent — v3 pipeline + lightweight enhancements.

Pipeline per round:
  Step A: code-forced search (BM25 top-15 → ReRanker → top-5)
  Step B: model screens results → picks ≤2 docids to read
  Step C: code-forced get_document
  Step D: model extracts facts + dead ends
  Step E: model audits constraints → NO (keep searching) / YES (answer ready)

Kept from Phase 1: ReRanker, dead-loop guard, sliding-window dedupe.
Removed: Planner phase 0, blind fetch fallback, context isolation, per-agent prompts.
"""

import json
import re
from typing import Any, Dict, List, Tuple


# ── Memory ──────────────────────────────────────────────────────────

class AgentMemory:
    def __init__(self):
        self.facts: List[Dict] = []       # {fact, docid, round}
        self.searched_queries: List[str] = []
        self.ruled_out: List[str] = []
        self.read_docids: List[str] = []
        self.fetched_docids: Dict[str, int] = {}
        self.last_assess: str = ""
        self._fact_keys: set = set()

    def add_fact(self, fact: str, docid: str = "", round_num: int = 0) -> bool:
        key = fact.strip().lower()
        if not key or key in self._fact_keys:
            return False
        self._fact_keys.add(key)
        self.facts.append({"fact": fact.strip(), "docid": docid, "round": round_num})
        return True

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

    def facts_summary(self) -> str:
        if not self.facts:
            return "(no facts yet)"
        lines = []
        for f in self.facts[-15:]:  # show last 15
            src = f" [doc {f['docid']}]" if f.get("docid") else ""
            lines.append(f"- {f['fact']}{src}")
        return "\n".join(lines)

    def searched_summary(self) -> str:
        if not self.searched_queries:
            return "(none)"
        return "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(self.searched_queries))

    def ruled_out_summary(self) -> str:
        if not self.ruled_out:
            return "(none)"
        return "\n".join(f"- {r}" for r in self.ruled_out)


# ── Retrieval ───────────────────────────────────────────────────────

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


# ── v3 Prompts ──────────────────────────────────────────────────────

SYSTEM_PROMPT = "Research agent. Find facts, answer questions. Output the requested format only."
SYSTEM_SCREEN = "Screen agent. Select relevant docs or output NONE."
SYSTEM_EXECUTOR = "Executor agent. Extract stated facts from documents. Do not re-extract ruled-out candidates."
SYSTEM_ASSESSOR = "Assessor agent. Audit constraints. YES if all satisfied, else NO + keywords."
SYSTEM_SYNTHESIZER = "Synthesizer agent. Answer from facts only. Do not fabricate."

DOC_SCREEN_PROMPT = """Pick at most 2 relevant documents. If none, output NONE.

Relevant DocIDs: ..."""

FACT_EXTRACT_PROMPT = """Extract facts from these documents that help answer the question. Skip candidates already listed in Ruled Out.

Facts Found:
- ...
(None if nothing)

Dead Ends:
- candidate: why ruled out
(None if nothing)"""

PROGRESS_PROMPT = """Audit each constraint from the question. BM25 matches exact words only.

Constraint Audit:
- constraint: satisfied / no evidence

Status: YES | NO

Search Query: ..."""

FINAL_ANSWER_PROMPT = "If the facts above are sufficient to answer, write the answer. If not, write: Unable to determine.\n\nAnswer:"

RETHINK_PROMPT = "New search angle. Search Query: ..."


# ── Helpers ─────────────────────────────────────────────────────────

def _chat(client, model, msgs, max_tok=2048):
    try:
        r = client.simple_chat(model=model, messages=msgs,
                               temperature=0.0, max_tokens=max_tok)
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _strip_think(text: str) -> str:
    return text


def _parse_docids(text: str, results: List[Dict]) -> List[str]:
    m = re.search(r'Relevant DocIDs:\s*', text, re.IGNORECASE)
    if not m:
        return []
    rest = text[m.end():]
    stop = ["Status:", "Next Query:", "Reasoning:", "Key Facts:",
            "Explanation:", "Confidence:", "Thought:", "Action:",
            "Facts Found:", "Dead Ends:", "Constraint Audit:", "Exact Answer:",
            "New Direction:"]
    content = []
    for line in rest.split("\n"):
        s = line.strip()
        if not s:
            break
        if any(s.startswith(p) for p in stop):
            break
        content.append(s)
    c = " ".join(content).strip()
    if not c or c.upper() == "NONE":
        return []

    result_docids = [d['docid'] for d in results]
    mapped, seen = [], set()
    for rid in result_docids:
        if rid in c and rid not in seen:
            seen.add(rid)
            mapped.append(rid)
            if len(mapped) >= 2:
                return mapped
    for d in re.findall(r'\b(\d+)\b', c):
        if d.isdigit() and 1 <= int(d) <= len(results):
            actual = results[int(d) - 1]['docid']
            if actual not in seen:
                seen.add(actual)
                mapped.append(actual)
            if len(mapped) >= 2:
                break
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
            # Filter placeholder and "none" variants
            clean = []
            for item in f:
                low = item.lower().strip()
                if low in ("none", "none.", "(none)", "(none if nothing)", "...", ""):
                    continue
                clean.append(item)
            return clean
    return []


def _parse_dead_ends(text: str) -> List[str]:
    d = _parse_section(text, "Dead Ends")
    clean = []
    for item in d:
        low = item.lower().strip()
        if low in ("none", "none.", "(none)", "(none if nothing)", "...", ""):
            continue
        clean.append(item)
    return clean


def _parse_status(text: str) -> str:
    m = re.search(r'Status:\s*\*{0,2}\s*(YES|NO)\s*\*{0,2}',
                  text, re.I)
    return m.group(1).upper() if m else "NO"


def _parse_search_query(text: str) -> str:
    for pat in [r'Search Query:\s*"?(.+?)"?\s*$',
                r'Search Query:\s*(.+?)\s*$']:
        m = re.search(pat, text, re.I | re.M)
        if m:
            q = m.group(1).strip().strip('"\'')
            if q and '...' not in q and '<' not in q:
                return q
    return ""


def _fallback_query(question: str, memory: AgentMemory) -> str:
    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'from', 'by', 'with',
            'and', 'or', 'but', 'not', 'this', 'that', 'these', 'those',
            'can', 'you', 'tell', 'find', 'what', 'when', 'where', 'who',
            'how', 'why', 'which', 'name', 'one', 'first', 'last', 'mid',
            'there', 'their', 'they', 'them', 'has', 'have', 'had', 'its',
            'also'}
    candidates = []
    candidates.extend(re.findall(r'"([^"]+)"', question))
    candidates.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                                 question))
    candidates.extend(re.findall(r'\b\d{3,4}\b', question))
    candidates.extend(re.findall(r'\b\d{1,2}(?:st|nd|rd|th)\b', question))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
    for i in range(len(words) - 1):
        w1, w2 = words[i].lower(), words[i+1].lower()
        if w1 not in stop or w2 not in stop:  # at least one meaningful word
            candidates.append(f"{words[i]} {words[i+1]}")

    seen = set()
    distinctive = []
    for c in candidates:
        cl = c.lower().strip()
        if cl not in stop and len(cl) > 2 and not cl.isdigit() and cl not in seen:
            seen.add(cl)
            distinctive.append(cl)

    out = []
    for t in distinctive:
        # Filter out pure-digit sequences and placeholders
        if t.isdigit() or all(w.isdigit() for w in t.split()):
            continue
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


def _format_docs(docs: List[Dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", doc.get("error", ""))
        if len(text) > 6000:
            text = text[:6000] + "\n... [truncated]"
        lines.append(f"--- Document {i} (docid={doc['docid']}) ---\n{text}")
    return "\n\n".join(lines) if lines else "(no documents)"


class ResearchContext:
    """Per-agent context — each step sees only what it needs (multi-agent via context isolation)."""

    def __init__(self, question: str, memory: AgentMemory):
        self.q = question
        self.m = memory

    def _trim(self, parts: list) -> str:
        result = "\n".join(parts)
        facts = self.m.facts
        while len(result) > 20000 and len(facts) > 3:
            facts = facts[2:]
            idx = next(i for i, p in enumerate(parts) if "## Facts" in p)
            summary = "\n".join(f"- {f['fact']}" for f in facts) if facts else "(none)"
            parts[idx] = f"\n## Facts\n{summary}"
            result = "\n".join(parts)
        return result

    def for_screen(self) -> str:
        return self._trim([f"## Question\n{self.q}",
                           f"\n## Facts\n{self.m.facts_summary()}"])

    def for_executor(self) -> str:
        return self._trim([f"## Question\n{self.q}",
                           f"\n## Facts\n{self.m.facts_summary()}",
                           f"\n## Ruled Out\n{self.m.ruled_out_summary()}"])

    def for_assessor(self, current_query: str = "") -> str:
        extra = (f"Most recent query: \"{current_query}\"\n"
                 f"Rounds searched: {len(self.m.searched_queries)}")
        return self._trim([
            f"## Question\n{self.q}",
            f"\n## Facts\n{self.m.facts_summary()}",
            f"\n## Ruled Out\n{self.m.ruled_out_summary()}",
            f"\n## Query History\n{self.m.searched_summary()}",
        ]) + f"\n\n{extra}"

    def for_synthesizer(self) -> str:
        parts = [f"## Question\n{self.q}",
                 f"\n## Facts\n{self.m.facts_summary()}"]
        result = self._trim(parts)
        if self.m.last_assess:
            audit = self.m.last_assess
            audit = re.sub(r'Status:.*$', '', audit, flags=re.M | re.I)
            audit = re.sub(r'Search Query:.*$', '', audit, flags=re.M | re.I)
            audit = audit.strip()
            if audit:
                result += f"\n\n## Constraint Audit\n{audit[:2000]}"
        return result


# ── Pipeline steps (multi-agent via ResearchContext) ───────────────

def _step_search(registry: Dict, query: str, memory: AgentMemory,
                 searcher=None) -> List[Dict]:
    memory.add_searched(query)
    search_fn = registry["search"]
    if searcher:
        try:
            raw = searcher.search(query, k=15)
        except Exception:
            raw = []
    else:
        try:
            raw = search_fn(query)
        except Exception:
            raw = []
    if not raw:
        return []
    return ReRanker.rerank(query, raw, top_k=5) if len(raw) > 5 else raw


def _step_screen(client, model, question: str, results: List[Dict],
                 memory: AgentMemory) -> str:
    ctx = ResearchContext(question, memory).for_screen()
    prompt = (f"{ctx}\n\n## Search Results\n{_format_results(results)}\n\n"
              f"{DOC_SCREEN_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_SCREEN},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=1024)


def _step_fetch(registry: Dict, docids: List[str],
                memory: AgentMemory) -> List[Dict]:
    get_doc_fn = registry["get_document"]
    docs = []
    for did in docids[:3]:
        fc = memory.fetched_docids.get(did, 0)
        if fc >= 2:
            continue
        try:
            doc = get_doc_fn(did)
        except Exception:
            doc = None
        if doc is None or (isinstance(doc, dict) and "error" in doc):
            docs.append({"docid": did, "error": "not found"})
        else:
            memory.add_read(did)
            memory.fetched_docids[did] = fc + 1
            docs.append({"docid": did, "text": doc.get("text", ""),
                         "url": doc.get("url", "")})
    return docs


def _step_extract(client, model, question: str, docs: List[Dict],
                  memory: AgentMemory) -> str:
    if not docs:
        return ""
    ctx = ResearchContext(question, memory).for_executor()
    prompt = (f"{ctx}\n\n## Full Documents\n{_format_docs(docs)}\n\n"
              f"{FACT_EXTRACT_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_EXECUTOR},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def _step_assess(client, model, question: str, memory: AgentMemory,
                 current_query: str) -> str:
    ctx = ResearchContext(question, memory).for_assessor(current_query)
    prompt = f"{ctx}\n\n{PROGRESS_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_ASSESSOR},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def _step_final_answer(client, model, question: str,
                       memory: AgentMemory) -> str:
    ctx = ResearchContext(question, memory).for_synthesizer()
    prompt = f"{ctx}\n\n{FINAL_ANSWER_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_SYNTHESIZER},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def _step_rethink(client, model, question: str,
                  memory: AgentMemory) -> str:
    ctx = ResearchContext(question, memory).for_assessor()
    prompt = f"{ctx}\n\n{RETHINK_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_ASSESSOR},
            {"role": "user", "content": prompt}]
    raw = _chat(client, model, msgs, max_tok=1024)
    return _parse_search_query(raw)


# ── Trajectory builder ──────────────────────────────────────────────

def _build_trajectory(question: str, memory: AgentMemory,
                      round_records: List[Dict],
                      final_answer: str) -> List[Dict]:
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    call_id = 0
    for rec in round_records:
        call_id += 1
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                            "function": {"name": "search",
                                         "arguments": json.dumps(
                                             {"query": rec["query"]},
                                             ensure_ascii=False)}}]})
        messages.append({
            "role": "tool", "tool_call_id": f"call_{call_id}",
            "content": json.dumps([
                {"docid": d["docid"], "score": d.get("score", 0),
                 "snippet": (d.get("text", "") or "")[:300]}
                for d in rec.get("results", [])
            ], ensure_ascii=False)})
        messages.append({"role": "assistant", "content": rec.get("screen", "")})
        fetched = rec.get("fetched")
        for d in (fetched or []):
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{call_id}", "type": "function",
                                "function": {"name": "get_document",
                                             "arguments": json.dumps(
                                                 {"docid": d["docid"]},
                                                 ensure_ascii=False)}}]})
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps(
                    {"docid": d["docid"],
                     "text_preview": (d.get("text", "") or "")[:500],
                     "url": d.get("url", "")},
                    ensure_ascii=False)})
        messages.append({"role": "assistant", "content": rec.get("extract", "")})
        messages.append({"role": "assistant", "content": rec.get("assess", "")})
    messages.append({"role": "assistant", "content": final_answer})
    return messages


# ── Main entry ──────────────────────────────────────────────────────

def run_agent_loop(
    client, model, query: str,
    tools: List[Dict], registry: Dict[str, Any],
    max_turns: int = 5, max_history_msgs: int = 6,
) -> Tuple[str, List[Dict]]:
    """v3-style 5-step pipeline + ReRanker + dead-loop guard."""

    print(f"[start] {query[:80]}...", flush=True)
    memory = AgentMemory()
    round_records: List[Dict] = []
    empty_rounds = 0
    suggested_query = ""
    searcher = registry.get("_searcher")

    for round_num in range(1, max_turns + 1):
        # ── Determine query (v3 style: R1 = full question) ──
        if round_num == 1:
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
            fb = _fallback_query(query, memory)
            if fb and not memory.was_recent(fb):
                current_query = fb
            else:
                break

        # Rethink after 2+ empty rounds
        if round_num > 1 and empty_rounds >= 2:
            rethink_q = _step_rethink(client, model, query, memory)
            if rethink_q and not memory.was_recent(rethink_q):
                current_query = rethink_q

        rec: Dict = {"query": current_query}

        # ── Step A: Search ──
        results = _step_search(registry, current_query, memory, searcher)
        rec["results"] = results
        if not results:
            empty_rounds += 1
            continue

        # ── Step B: Screen (v3: no blind fetch — NONE means skip) ──
        screen_raw = _step_screen(client, model, query, results, memory)
        screen_raw = _strip_think(screen_raw)
        rec["screen"] = screen_raw
        docids = _parse_docids(screen_raw, results)

        # Fallback: if Screen output is all think with no docids, take top-1
        if not docids and results and '<think>' in screen_raw:
            docids = [results[0]["docid"]]

        # ── Step C: Fetch ──
        docs = _step_fetch(registry, docids, memory) if docids else []
        rec["fetched"] = docs

        # ── Step D: Extract facts ──
        extract_raw = _step_extract(client, model, query, docs, memory) if docs else ""
        extract_raw = _strip_think(extract_raw)
        rec["extract"] = extract_raw
        new_facts = _parse_facts(extract_raw)
        dead_ends = _parse_dead_ends(extract_raw)
        # Store facts with source docid and round number
        src_docid = docs[0]["docid"] if docs else ""
        for f in new_facts:
            memory.add_fact(f, docid=src_docid, round_num=round_num)
        for d in dead_ends:
            memory.add_ruled_out(d)

        if new_facts:
            empty_rounds = 0
        else:
            empty_rounds += 1

        # ── Step E: Assess ──
        assess_raw = _step_assess(client, model, query, memory, current_query)
        assess_raw = _strip_think(assess_raw)
        rec["assess"] = assess_raw
        memory.last_assess = assess_raw

        round_records.append(rec)
        status = _parse_status(assess_raw)

        if status == "YES":
            break

        # Prepare next query (filter placeholders)
        next_q = _parse_search_query(assess_raw)
        if next_q and ('<' in next_q or len(next_q.split()) < 2):
            next_q = ""  # reject placeholders and single words
        if not next_q:
            fb = _fallback_query(query, memory)
            if fb and not memory.was_recent(fb):
                next_q = fb
            else:
                break
        suggested_query = next_q

    # ── Final answer ──
    answer_raw = _step_final_answer(client, model, query, memory)
    # Try Answer: or Exact Answer: format
    m = re.search(r'(?:Exact )?Answer:\s*(.+?)(?:\n|$)', answer_raw, re.IGNORECASE)
    if m:
        final_answer = m.group(1).strip()
    else:
        # Fallback: strip think tags and take last meaningful line
        t = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL)
        t = re.sub(r'<think>.*$', '', t, flags=re.DOTALL).strip()
        if t:
            final_answer = t
        else:
            # Everything was in think tags — extract inner content
            inner = re.findall(r'<think>(.*?)</think>', answer_raw, re.DOTALL)
            final_answer = '\n'.join(s.strip() for s in inner if s.strip()) if inner else answer_raw.strip()
    # Defensive: if Synthesizer leaked assessor format, reject
    final_answer = re.sub(r'^Status:\s*(YES|NO)\s*', '', final_answer).strip()
    if final_answer.strip().upper() in ('YES', 'NO', ''):
        final_answer = 'Unable to determine from available evidence.'
    if final_answer.startswith('ERROR'):
        final_answer = 'Unable to determine from available evidence.'

    trajectory = _build_trajectory(query, memory, round_records, final_answer)
    return final_answer, trajectory