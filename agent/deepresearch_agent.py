# -*- coding: utf-8 -*-
"""
Deep Research Agent — Phase 1 refactored (single-file).

Key improvements over original:
  - EvidenceStore: structured {fact, docid, confidence, source_quote} vs List[str]
  - QueryHistory: sliding-window dedupe (window=2) vs permanent Mem.seen()
  - Executor: JSON structured output vs natural language paragraphs
  - Two-stage retrieval: BM25 top-15 → ReRanker → top-5
  - QueryAwareChunker: query-relevant chunk selection vs text[:5000]
  - Real tool observations in trajectory
"""

import json
import re
from typing import Any, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════
# Memory
# ══════════════════════════════════════════════════════════════════════

class EvidenceStore:
    """Structured evidence storage — replaces plain List[str] findings."""

    def __init__(self):
        self._items: List[Dict] = []
        self._fact_set: set = set()
        self._ruled_out: List[Dict] = []

    def add(self, fact: str, docid: str = "", query: str = "",
            round_num: int = 0, confidence: str = "medium",
            source_quote: str = "") -> bool:
        key = fact.strip().lower()
        if not key or key in self._fact_set:
            return False
        self._fact_set.add(key)
        self._items.append(dict(fact=fact.strip(), docid=docid, query=query,
                                round=round_num, confidence=confidence,
                                source_quote=source_quote))
        return True

    def add_batch(self, evidence_list: List[Dict],
                  query: str = "", round_num: int = 0) -> int:
        added = 0
        for e in evidence_list:
            if isinstance(e, dict) and "fact" in e:
                if self.add(e["fact"], e.get("docid", ""), query, round_num,
                            e.get("confidence", "medium"), e.get("source_quote", "")):
                    added += 1
        return added

    def get_all(self) -> List[Dict]:
        return list(self._items)

    def get_high_confidence(self) -> List[Dict]:
        return [e for e in self._items if e["confidence"] == "high"]

    def has_fact(self, fact: str) -> bool:
        return fact.strip().lower() in self._fact_set

    def count(self) -> int:
        return len(self._items)

    def add_ruled_out(self, candidate: str, reason: str = ""):
        c = candidate.strip()
        if c and c not in {r["candidate"] for r in self._ruled_out}:
            self._ruled_out.append({"candidate": c, "reason": reason})

    def summary_for_context(self, max_items: int = 10) -> str:
        if not self._items:
            return "(no evidence collected yet)"
        recent = self._items[-max_items:]
        lines = []
        for e in recent:
            src = f" (doc: {e['docid']})" if e.get("docid") else ""
            lines.append(f"- [{e['confidence']}]{src} {e['fact']}")
        return "\n".join(lines)

    def full_summary(self) -> str:
        parts = [f"## Collected Evidence ({self.count()} items)"]
        parts.append(self.summary_for_context(max_items=50))
        if self._ruled_out:
            parts.append("\n## Ruled Out\n" + "\n".join(
                f"- {r['candidate']}: {r['reason']}"
                for r in self._ruled_out[-5:]))
        return "\n".join(parts)


class QueryHistory:
    """Sliding-window query dedupe — replaces permanent Mem.seen()."""

    def __init__(self, window_size: int = 2):
        self._queries: List[str] = []
        self._window_size = window_size

    def add(self, query: str):
        q = query.strip()
        if q:
            self._queries.append(q)

    def was_recent(self, query: str) -> bool:
        """Only blocks repeats in last window_size entries. A→B→A passes."""
        q = query.strip().lower()
        if not q:
            return False
        return any(q == r.lower() for r in self._queries[-self._window_size:])

    def was_ever_searched(self, query: str) -> bool:
        q = query.strip().lower()
        return any(q == prev.lower() for prev in self._queries)

    def all_queries(self) -> List[str]:
        return list(self._queries)

    def recent_queries(self, n: int = 5) -> List[str]:
        return self._queries[-n:]

    def count(self) -> int:
        return len(self._queries)


# ══════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════

class ReRanker:
    """Two-stage retrieval: BM25 top-15 → re-rank to top-5 via term overlap."""

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
        d_terms = set(text.lower().split())
        return len(q_terms & d_terms) / len(q_terms)

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
    """Query-aware document chunking — replaces naive text[:5000]."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
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
    def select_relevant_chunks(cls, query: str, text: str,
                               max_chunks: int = 5,
                               chunk_size: int = 1000,
                               overlap: int = 200) -> str:
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
                          max_total_chars: int = 5000) -> str:
        max_chunks = max(1, max_total_chars // 1000)
        result = cls.select_relevant_chunks(query, text, max_chunks=max_chunks)
        return result[:max_total_chars] if len(result) > max_total_chars else result


# ══════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════

PLANNER_SYS = "Break complex questions into searchable sub-queries. Output one per line with '- ' prefix."

PLANNER_PROMPT = """Decompose this question into 3-5 search queries. Each query must include specific entities, dates, and details from the question. Output ONLY lines starting with '- '.

Example:
Question: A restaurant founded in the 1950s in Chicago by a chef trained in France in the 1940s, with a signature dish technique on pages 50-55 of their cookbook. The chef's mentor worked at a Paris hotel in the 1920s. Name the first executive pastry chef.

Output:
- 1950s Chicago restaurant French-trained chef 1940s
- signature dish cooking technique pages 50-55 cookbook
- chef mentor Paris hotel 1920s
- first executive pastry chef restaurant name"""

EXECUTOR_SYS = (
    "You are a precise evidence extraction agent. "
    "Extract specific, verifiable facts from documents as structured JSON. "
    "Only include facts directly stated in the documents. Do not infer."
)

EXECUTOR_PROMPT = """## Question
{question}

## Documents
{documents}

## Task
Extract ALL facts relevant to answering the question. Include supporting quotes.

Output ONLY a JSON array (no markdown fences, no other text):
[
  {{
    "fact": "specific verifiable fact",
    "docid": "source document ID",
    "confidence": "high",
    "source_quote": "exact supporting text from the document"
  }}
]

Confidence: "high" (explicitly stated, exact match), "medium" (stated but some ambiguity), "low" (hinted/partial).
If nothing relevant: output []"""

ASSESS_SYS = "Audit research progress. Check what is known, what is missing, and suggest next queries."

ASSESS_PROMPT = """## Question
{question}

{evidence_summary}

## Query History
{query_history}

## Task
Audit progress. Output in this exact format:

Status: NEED_MORE | READY_TO_ANSWER
Known Facts:
- fact (source: docid)
Missing Information:
- specific detail still needed
Next Queries:
- keyword query 1
- keyword query 2

If all constraints are satisfied with high-confidence evidence, output READY_TO_ANSWER.
If information is still missing, output NEED_MORE with specific next queries (3-6 distinctive keywords each)."""

RETHINK_PROMPT = """The query "{last_query}" found nothing useful.

Design a completely different search query — different angle, different keywords.

Output exactly:
Search Query: <3-5 keywords>"""

SYNTHESIZER_SYS = "Answer the question based strictly on the collected evidence. Do not fabricate."

SYNTHESIZER_PROMPT = """## Question
{question}

{evidence_summary}

## Task
Based on the evidence above, what is the answer? Be precise.

Output exactly:
Exact Answer: <answer>

If evidence is insufficient, output:
Exact Answer: Unable to determine from available evidence."""


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


def _strip(text):
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return re.sub(r'<think>.*', '', t, flags=re.DOTALL).strip() or text.strip()


def _parse_queries(text: str) -> List[str]:
    queries = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('-'):
            q = s[1:].strip().strip('"\'')
            q = q.replace('**', '').replace('*', '').strip()
            if len(q) >= 10:
                queries.append(q)
    return queries


def _parse_section(text: str, heading: str) -> List[str]:
    items, in_block = [], False
    pat = re.compile(re.escape(heading), re.I)
    for line in text.split('\n'):
        s = line.strip()
        if pat.search(s):
            in_block = True
            continue
        if in_block and s.startswith('-'):
            items.append(s[1:].strip())
        elif in_block and s and not s.startswith('-'):
            break
    return items


def _parse_json_output(raw: str) -> List[Dict]:
    cleaned = _strip(raw)
    fence = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    arr = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if not arr:
        return []
    try:
        parsed = json.loads(arr.group(0))
        if isinstance(parsed, list):
            return [item for item in parsed
                    if isinstance(item, dict) and "fact" in item]
    except json.JSONDecodeError:
        pass
    return []


def _fmt_docs(docs: List[Dict], query: str = "") -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        text = d.get("text", d.get("error", ""))
        chunked = QueryAwareChunker.chunk_for_context(
            query if query else "", text, max_total_chars=5000)
        parts.append(f"--- Doc {i} (docid={d['docid']}, "
                     f"url={d.get('url', '')}) ---\n{chunked}")
    return "\n\n".join(parts) if parts else "(no documents)"


def _fallback_queries(question: str, qh: QueryHistory) -> List[str]:
    cand = []
    cand.extend(re.findall(r'"([^"]+)"', question))
    cand.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question))
    cand.extend(re.findall(r'\b\d{3,4}\b', question))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
    for i in range(len(words) - 1):
        cand.append(f"{words[i]} {words[i+1]}")
    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'from', 'by', 'with',
            'and', 'or', 'but', 'not', 'this', 'that', 'these', 'those',
            'can', 'you', 'tell', 'find', 'what', 'when', 'where', 'who',
            'how', 'why', 'which', 'name', 'one', 'first', 'last', 'mid',
            'there', 'their', 'they', 'them', 'has', 'have', 'had', 'its',
            'also'}
    dist = []
    for c in cand:
        cl = c.lower().strip()
        if cl not in stop and len(cl) > 2:
            dist.append(cl)
    out = []
    for t in dist:
        if not qh.was_recent(t) and t not in out:
            out.append(t)
        if len(out) >= 5:
            break
    return out


# ══════════════════════════════════════════════════════════════════════
# Agent functions
# ══════════════════════════════════════════════════════════════════════

def _plan(client, model, question: str, extra_context: str = "") -> List[str]:
    user = f"## Question\n{question}\n\n{PLANNER_PROMPT}"
    if extra_context:
        user = (f"## Question\n{question}\n\n"
                f"## What We Already Know\n{extra_context}\n\n"
                f"{PLANNER_PROMPT}\nFocus on what is still MISSING.")
    msgs = [{"role": "system", "content": PLANNER_SYS},
            {"role": "user", "content": user}]
    raw = _strip(_chat(client, model, msgs, max_tok=512))
    queries = _parse_queries(raw)
    return queries if queries else [question]


def _execute(client, model, question: str,
             docs: List[Dict], search_query: str = "") -> List[Dict]:
    if not docs:
        return []
    msgs = [
        {"role": "system", "content": EXECUTOR_SYS},
        {"role": "user", "content": EXECUTOR_PROMPT.format(
            question=question, documents=_fmt_docs(docs, search_query))},
    ]
    raw = _chat(client, model, msgs, max_tok=2048)
    return _parse_json_output(raw)


def _assess(client, model, question: str,
            evidence: EvidenceStore, qh: QueryHistory) -> Dict:
    queries_text = "\n".join(
        f"  [{i+1}] {q}" for i, q in enumerate(qh.recent_queries(8))
    ) or "(none)"

    ctx = ASSESS_PROMPT.format(
        question=question,
        evidence_summary=evidence.full_summary(),
        query_history=queries_text,
    )
    msgs = [{"role": "system", "content": ASSESS_SYS},
            {"role": "user", "content": ctx}]
    raw = _strip(_chat(client, model, msgs, max_tok=1024))

    status = "READY_TO_ANSWER" if re.search(
        r'\bREADY_TO_ANSWER\b', raw, re.I) else "NEED_MORE"
    known = _parse_section(raw, "Known Facts")
    missing = _parse_section(raw, "Missing Information")
    next_queries = _parse_section(raw, "Next Queries")

    if not next_queries and status == "NEED_MORE":
        for m in re.finditer(
            r'Search Query:\s*"?(.+?)"?\s*$', raw, re.I | re.M):
            q = m.group(1).strip().strip('"\'')
            q = q.replace('**', '').replace('*', '').strip()
            if q:
                next_queries.append(q)

    return {"status": status, "known_facts": known,
            "missing": missing, "next_queries": next_queries}


def _synthesize(client, model, question: str,
                evidence: EvidenceStore) -> str:
    ctx = SYNTHESIZER_PROMPT.format(
        question=question, evidence_summary=evidence.full_summary())
    msgs = [{"role": "system", "content": SYNTHESIZER_SYS},
            {"role": "user", "content": ctx}]
    raw = _strip(_chat(client, model, msgs, max_tok=512))
    m = re.search(r'Exact Answer:\s*(.+)', raw, re.I)
    if m:
        return m.group(1).strip()
    for line in raw.split('\n'):
        s = line.strip()
        if s and len(s) > 5 and not s.startswith(('Exact Answer:', 'ERROR:')):
            return s
    return raw.strip() or "Unable to determine answer from available evidence."


def _rethink(client, model, question: str, evidence: EvidenceStore,
             qh: QueryHistory, last_query: str) -> str:
    queries_text = "\n".join(
        f"  [{i+1}] {q}" for i, q in enumerate(qh.recent_queries(8))
    ) or "(none)"

    ctx = (f"## Question\n{question}\n\n{evidence.full_summary()}\n\n"
           f"## Query History\n{queries_text}\n\n"
           f"{RETHINK_PROMPT.replace('{last_query}', last_query)}")
    msgs = [
        {"role": "system",
         "content": "Fix failed search queries by designing new directions."},
        {"role": "user", "content": ctx},
    ]
    try:
        r = client.simple_chat(model=model, messages=msgs,
                               temperature=0.0, max_tokens=512)
        raw = r["choices"][0]["message"]["content"]
    except Exception:
        return ""
    raw = _strip(raw)
    m = re.search(r'Search Query:\s*"?(.+?)"?\s*$', raw, re.I | re.M)
    return m.group(1).strip().strip('"\'') if m else ""


# ══════════════════════════════════════════════════════════════════════
# Trajectory builder
# ══════════════════════════════════════════════════════════════════════

def _build_trajectory(question: str, evidence: EvidenceStore,
                      qh: QueryHistory, round_records: List[Dict],
                      final_answer: str) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "Deep research agent — structured evidence pipeline."},
        {"role": "user", "content": question},
    ]
    cid = 0
    for rec in round_records:
        for q in rec.get("queries", []):
            cid += 1
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{cid}", "type": "function",
                    "function": {"name": "search",
                                 "arguments": json.dumps({"query": q},
                                                         ensure_ascii=False)}}]})
            results = rec.get("results", [])
            msgs.append({
                "role": "tool", "tool_call_id": f"call_{cid}",
                "content": json.dumps([
                    {"docid": r["docid"], "score": r.get("score", 0),
                     "snippet": (r.get("text", "") or "")[:300]}
                    for r in results], ensure_ascii=False)})

        fetched = rec.get("fetched", [])
        if fetched:
            cid += 1
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{cid}", "type": "function",
                    "function": {"name": "get_document",
                                 "arguments": json.dumps({"docid": d["docid"]},
                                                         ensure_ascii=False)}}
                    for d in fetched]})
            msgs.append({
                "role": "tool", "tool_call_id": f"call_{cid}",
                "content": json.dumps([
                    {"docid": d["docid"],
                     "text_preview": (d.get("text", "") or "")[:500],
                     "url": d.get("url", "")}
                    for d in fetched], ensure_ascii=False)})

        if rec.get("extract"):
            msgs.append({"role": "assistant", "content": rec["extract"]})
        if rec.get("assess"):
            msgs.append({"role": "assistant", "content": rec["assess"]})

    msgs.append({"role": "assistant", "content": final_answer})
    return msgs


# ══════════════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════════════

def run_agent_loop(
    client, model, query: str,
    tools: List[Dict], registry: Dict[str, Any],
    max_turns: int = 5, max_history_msgs: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    search_fn = registry["search"]
    get_doc_fn = registry["get_document"]
    searcher = registry.get("_searcher")

    evidence = EvidenceStore()
    qh = QueryHistory(window_size=2)
    round_records: List[Dict] = []
    final_answer = ""
    empty_streak = 0

    all_queries = _plan(client, model, query)
    if not all_queries:
        all_queries = [query]

    for rnd in range(1, max_turns + 1):
        rec: Dict[str, Any] = {"queries": list(all_queries)}
        all_docs: List[Dict] = []
        all_results: List[Dict] = []

        # ── Search + Rerank + Fetch ──
        for q in all_queries:
            if not q or not q.strip():
                continue
            q = q.strip()
            if qh.was_recent(q):
                continue
            qh.add(q)

            if searcher:
                try:
                    raw_results = searcher.search(q, k=15)
                except Exception:
                    raw_results = []
            else:
                try:
                    raw_results = search_fn(q)
                except Exception:
                    raw_results = []

            if not raw_results:
                continue

            reranked = ReRanker.rerank(q, raw_results, top_k=5)
            all_results.extend(reranked)

            for r in reranked[:3]:
                did = r.get("docid")
                if not did:
                    continue
                try:
                    doc = get_doc_fn(did)
                except Exception:
                    continue
                if doc and "error" not in doc:
                    all_docs.append({
                        "docid": did, "text": doc.get("text", ""),
                        "url": doc.get("url", "")})

        rec["results"] = all_results
        rec["fetched"] = all_docs

        if not all_docs:
            empty_streak += 1
            last_q = all_queries[0] if all_queries else ""
            if last_q:
                new_q = _rethink(client, model, query, evidence, qh, last_q)
                if new_q and not qh.was_recent(new_q):
                    all_queries = [new_q]
                else:
                    fb = _fallback_queries(query, qh)
                    all_queries = fb if fb else []
            else:
                fb = _fallback_queries(query, qh)
                all_queries = fb if fb else []

            if not all_queries:
                break
            continue

        # ── Execute ──
        search_q = all_queries[0] if all_queries else query
        evidence_list = _execute(client, model, query, all_docs, search_q)
        added = evidence.add_batch(evidence_list, query=search_q, round_num=rnd)
        rec["extract"] = json.dumps(evidence_list, ensure_ascii=False)

        if added == 0:
            empty_streak += 1
        else:
            empty_streak = 0

        # ── Assess ──
        assess_result = _assess(client, model, query, evidence, qh)
        rec["assess"] = (
            f"Status: {assess_result['status']}\n"
            + (f"Known: {'; '.join(assess_result['known_facts'][:5])}\n"
               if assess_result.get('known_facts') else "")
            + (f"Missing: {'; '.join(assess_result['missing'][:5])}\n"
               if assess_result.get('missing') else "")
            + (f"Next Queries: {', '.join(assess_result['next_queries'][:5])}"
               if assess_result.get('next_queries') else ""))
        round_records.append(rec)

        if assess_result["status"] == "READY_TO_ANSWER":
            break

        # ── Next queries (multi-query) ──
        next_queries = assess_result.get("next_queries", [])
        if next_queries:
            filtered = [q for q in next_queries if not qh.was_recent(q)]
            if filtered:
                all_queries = filtered
                continue

        # ── Stuck: fallback chain ──
        if empty_streak >= 2:
            fb = _fallback_queries(query, qh)
            if fb:
                all_queries = fb
                continue

        last_q = next_queries[0] if next_queries else (
            all_queries[0] if all_queries else "")
        if last_q:
            new_q = _rethink(client, model, query, evidence, qh, last_q)
            if new_q and not qh.was_recent(new_q):
                all_queries = [new_q]
                continue

        ctx = f"Findings so far: {evidence.summary_for_context(max_items=5)}"
        new_queries = _plan(client, model, query, ctx)
        if new_queries:
            all_queries = [q for q in new_queries if not qh.was_recent(q)]
            if all_queries:
                continue

        break

    # ── Synthesize ──
    final_answer = _synthesize(client, model, query, evidence)
    if not final_answer or final_answer.startswith("ERROR"):
        final_answer = "Unable to determine answer from available evidence."

    trajectory = _build_trajectory(query, evidence, qh, round_records, final_answer)
    return final_answer, trajectory
