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
        self.last_assess: str = ""  # for Synthesizer context

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

SYSTEM_PLANNER = (
    "You decompose complex questions into BM25 keyword queries across multiple angles. "
    "BM25 does pure keyword matching — no semantics, no synonyms. "
    "Only exact word overlap counts. Rare distinctive words dominate."
)

SYSTEM_SCREEN = (
    "You screen search results and decide which documents are worth reading in full. "
    "Select only documents whose snippets show direct relevance to the question. "
    "Do not guess — if nothing looks relevant, output NONE."
)

SYSTEM_EXECUTOR = (
    "You extract specific verifiable facts from documents. "
    "Only report information directly stated in the documents. "
    "Do not infer, extrapolate, or hallucinate. "
    "If nothing is relevant, honestly say None."
)

SYSTEM_ASSESSOR = (
    "You audit research progress against every constraint in the question. "
    "Be rigorous: check each constraint independently. "
    "If any constraint lacks evidence, say NEED_MORE and suggest a new keyword query. "
    "Only say READY_TO_ANSWER when ALL constraints are satisfied by confirmed facts."
)

SYSTEM_RETHINK = (
    "You are stuck — previous searches found nothing. "
    "Find a genuinely new search direction using different keywords, "
    "a different entity, or a different angle than what was already tried."
)

SYSTEM_SYNTHESIZER = (
    "You produce the final answer based strictly on collected evidence. "
    "Base your answer only on confirmed facts. "
    "If evidence is insufficient, say so honestly. Do not fabricate."
)

PLANNER_PROMPT = """Decompose the question into 3-5 keyword search directions. Output one per line.

- Each line: 3-6 space-separated keywords (no sentences, no punctuation)
- Cover different angles: entities, dates, events
- Use distinctive words that appear verbatim in documents

Output:
- keyword1 keyword2 keyword3
- keyword4 keyword5 keyword6"""

SCREEN_PROMPT = """Pick at most 2 documents worth reading. Output:

Relevant DocIDs: <docid1, docid2 or NONE>"""

EXECUTOR_PROMPT = """Extract facts and dead ends from these documents.

Facts Found:
- specific fact from the document relevant to the question
- (write "None" if nothing relevant)

Dead Ends:
- candidate: why it violates a constraint
- (write "None" if nothing)"""

ASSESSOR_PROMPT = """Audit each constraint from the question against confirmed facts.

Constraint Audit:
- constraint: satisfied / no evidence

Status: NEED_MORE | READY_TO_ANSWER

If NEED_MORE:
Search Query: <3-6 keywords>"""

RETHINK_PROMPT = """Previous searches found nothing. Suggest a completely different search direction.

Search Query: <3-6 keywords>"""

SYNTHESIZER_PROMPT = """Answer based only on confirmed facts. If insufficient, say: Unable to determine.

Exact Answer: <answer>"""


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
    return text  # no-op: let parsers work on raw output including think tags


def _parse_queries(text: str) -> List[str]:
    queries = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('-'):
            q = s[1:].strip().strip('"\'')
            q = q.replace('**', '').replace('*', '').strip()
            words = q.split()
            # Reject queries with too many words (>10) or containing parentheses
            if not (2 <= len(words) <= 10):
                continue
            if '(' in q or ')' in q:
                continue
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

    result_docids = [d['docid'] for d in results]
    mapped, seen = [], set()

    # 1. Exact docid substring match (supports any docid format)
    for rid in result_docids:
        if rid in c:
            if rid not in seen:
                seen.add(rid)
                mapped.append(rid)
            if len(mapped) >= 2:
                return mapped

    # 2. Fallback: numeric index mapping
    num_docids = re.findall(r'\b(\d+)\b', c)
    for d in num_docids:
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


def _clean_items(items: List[str]) -> List[str]:
    ignore = {"none", "none.", ""}
    return [i for i in items if i.lower().strip(".") not in ignore]


def _parse_facts(text: str) -> List[str]:
    for h in ["Facts Found", "facts found", "Key Facts", "key facts"]:
        f = _parse_section(text, h)
        if f:
            return _clean_items(f)
    return []


def _parse_dead_ends(text: str) -> List[str]:
    d = _parse_section(text, "Dead Ends")
    return _clean_items(d)


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


class ResearchContext:
    """Per-agent context manager — each agent sees only what it needs."""

    def __init__(self, question: str, memory: AgentMemory):
        self.q = question
        self.m = memory

    def _trim_facts(self, parts: list, max_chars: int = 20000):
        """Trim oldest facts if context too long."""
        result = "\n".join(parts)
        facts = self.m.confirmed_facts
        while len(result) > max_chars and len(facts) > 3:
            facts = facts[2:]
            idx = next((i for i, p in enumerate(parts)
                        if p.startswith("\n## Confirmed Facts")), 1)
            parts[idx] = "\n## Confirmed Facts\n" + (
                "\n".join(f"- {f}" for f in facts) if facts else "(none)")
            result = "\n".join(parts)
        return result

    def for_planner(self, missing_info: str = "") -> str:
        """Planner sees: question + (optional) what's still missing."""
        parts = [f"## Question\n{self.q}"]
        if missing_info:
            parts.append(f"\n## Missing Information\n{missing_info}")
        return "\n".join(parts)

    def for_screen(self) -> str:
        """Screen sees: question + confirmed facts (no query history to avoid bias)."""
        parts = [
            f"## Question\n{self.q}",
            f"\n## Confirmed Facts\n{self.m.facts_summary()}",
        ]
        return self._trim_facts(parts)

    def for_executor(self) -> str:
        """Executor sees: question + facts + evidence + ruled out."""
        parts = [
            f"## Question\n{self.q}",
            f"\n## Confirmed Facts\n{self.m.facts_summary()}",
            f"\n## Ruled Out\n{self.m.ruled_out_summary()}",
        ]
        return self._trim_facts(parts)

    def for_assessor(self, current_query: str = "") -> str:
        """Assessor sees: everything — facts, evidence, ruled out, query history."""
        parts = [
            f"## Question\n{self.q}",
            f"\n## Confirmed Facts\n{self.m.facts_summary()}",
            f"\n## Ruled Out\n{self.m.ruled_out_summary()}",
            f"\n## Query History\n{self.m.searched_summary()}",
        ]
        extra = (f"Most recent query: \"{current_query}\"\n"
                 + f"Rounds searched: {len(self.m.searched_queries)}")
        return self._trim_facts(parts) + f"\n\n{extra}"

    def for_synthesizer(self) -> str:
        """Synthesizer sees: question + facts + evidence + assessor's final audit."""
        parts = [
            f"## Question\n{self.q}",
            f"\n## Confirmed Facts\n{self.m.facts_summary()}",
        ]
        if self.m.evidence:
            parts.append(f"\n## Structured Evidence\n{self.m.evidence_summary()}")
        result = self._trim_facts(parts)
        if self.m.last_assess:
            result += f"\n\n## Assessor's Final Audit\n{self.m.last_assess[:2000]}"
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
    msgs = [{"role": "system", "content": SYSTEM_PLANNER},
            {"role": "user", "content": user}]
    raw = _strip_think(_chat(client, model, msgs, max_tok=2048))
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
    ctx = ResearchContext(question, memory)
    prompt = (f"{ctx.for_screen()}\n\n## Search Results\n{_format_results(results)}\n\n"
              f"{SCREEN_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_SCREEN},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


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
    ctx = ResearchContext(question, memory)
    prompt = (f"{ctx.for_executor()}\n\n"
              f"## Full Documents\n{_format_docs(docs, search_query)}\n\n"
              f"{EXECUTOR_PROMPT}")
    msgs = [{"role": "system", "content": SYSTEM_EXECUTOR},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def agent_assess(client, model, question: str, memory: AgentMemory,
                 current_query: str) -> str:
    ctx = ResearchContext(question, memory)
    prompt = f"{ctx.for_assessor(current_query)}\n\n{ASSESSOR_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_ASSESSOR},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)


def agent_rethink(client, model, question: str,
                  memory: AgentMemory) -> str:
    ctx = ResearchContext(question, memory)
    prompt = f"{ctx.for_assessor()}\n\n{RETHINK_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_RETHINK},
            {"role": "user", "content": prompt}]
    raw = _chat(client, model, msgs, max_tok=2048)
    return _parse_search_query(raw)


def agent_synthesize(client, model, question: str,
                     memory: AgentMemory) -> str:
    ctx = ResearchContext(question, memory)
    prompt = f"{ctx.for_synthesizer()}\n\n{SYNTHESIZER_PROMPT}"
    msgs = [{"role": "system", "content": SYSTEM_SYNTHESIZER},
            {"role": "user", "content": prompt}]
    return _chat(client, model, msgs, max_tok=2048)

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
        for d in rec.get("fetched", []):
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

    print(f"[start] {query[:80]}...", flush=True)
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
            current_query = angle_pool[0] if angle_pool else ""
            if angle_pool:
                angle_idx = 1
        else:
            current_query = suggested_query

        # Empty guard + fallback
        if not current_query or not current_query.strip():
            current_query = _fallback_query(query, memory)

        # Dedup: if recent, try next angle or fallback
        if memory.was_recent(current_query):
            if angle_idx < len(angle_pool):
                current_query = angle_pool[angle_idx]
                angle_idx += 1
            else:
                current_query = _fallback_query(query, memory)
            if not current_query or memory.was_recent(current_query):
                break

        # Stuck handling: state-machine based on empty_rounds severity
        if round_num > 1 and empty_rounds >= 2:
            rethink_q = agent_rethink(client, model, query, memory)
            if rethink_q and not memory.was_recent(rethink_q):
                current_query = rethink_q
                memory.add_note("Forced rethink — new direction")
            else:
                break
        elif round_num > 1 and empty_rounds >= 1:
            if angle_idx < len(angle_pool) and not memory.was_recent(angle_pool[angle_idx]):
                current_query = angle_pool[angle_idx]
                angle_idx += 1
                memory.add_note(f"Switching angle {angle_idx}/{len(angle_pool)}")
            elif suggested_query and not memory.was_recent(suggested_query):
                current_query = suggested_query
            else:
                fb = _fallback_query(query, memory)
                if fb and not memory.was_recent(fb):
                    current_query = fb
                else:
                    break

        rec: Dict = {"queries": [current_query]}

        # ── Retrieve + Rerank ──
        results = agent_retrieve(registry, current_query, memory, searcher)
        rec["results"] = results
        if not results:
            empty_rounds += 1
            round_records.append(rec)
            continue

        # ── Screen: LLM picks docs ──
        screen_raw = agent_screen(client, model, query, results, memory)
        screen_raw = _strip_think(screen_raw)
        rec["screen"] = screen_raw
        docids = _parse_docids(screen_raw, results)

        # Fallback: if Screen produced no docids, blindly take top-2 by score
        if not docids and results:
            docids = [r["docid"] for r in results[:2]]

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
        memory.last_assess = assess_raw  # for Synthesizer

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
