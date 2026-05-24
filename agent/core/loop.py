"""Main orchestration loop — ties together Planner/Executor/Assess/Synthesizer."""

import json
import re
from typing import Any, Callable, Dict, List, Tuple

from ..agents.assessor import assess
from ..agents.executor import execute
from ..agents.planner import plan
from ..agents.synthesizer import synthesize
from ..memory.evidence_store import EvidenceStore
from ..memory.query_history import QueryHistory
from ..retrieval.reranker import ReRanker
from .trajectory import build_trajectory


# ── Fallback query generator ──────────────────────────────────────

def _fallback_queries(question: str, qh: QueryHistory) -> List[str]:
    """Regex-extract keyword queries from question. Skips queries that were
    recently searched (window=2), NOT all queries ever searched."""
    cand = []
    cand.extend(re.findall(r'"([^"]+)"', question))
    cand.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question))
    cand.extend(re.findall(r'\b\d{3,4}\b', question))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
    for i in range(len(words) - 1):
        cand.append(f"{words[i]} {words[i+1]}")

    stop = {'the','a','an','is','was','are','were','be','been','in','on','at',
            'to','for','of','from','by','with','and','or','but','not','this',
            'that','these','those','can','you','tell','find','what','when',
            'where','who','how','why','which','name','one','first','last','mid',
            'there','their','they','them','has','have','had','its','also'}
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


# ── Rethink ───────────────────────────────────────────────────────

RETHINK_PROMPT = """The query "{last_query}" found nothing useful.

Design a completely different search query — different angle, different keywords.

Output exactly:
Search Query: <3-5 keywords>"""


def _rethink(client, model, question: str, evidence_store: EvidenceStore,
             qh: QueryHistory, last_query: str) -> str:
    queries_text = "\n".join(
        f"  [{i+1}] {q}" for i, q in enumerate(qh.recent_queries(8))
    ) or "(none)"
    ctx = (
        f"## Question\n{question}\n\n"
        f"{evidence_store.full_summary()}\n\n"
        f"## Query History\n{queries_text}\n\n"
        f"{RETHINK_PROMPT.replace('{last_query}', last_query)}"
    )
    msgs = [
        {"role": "system", "content": "Fix failed search queries by designing new directions."},
        {"role": "user", "content": ctx},
    ]
    try:
        r = client.simple_chat(model=model, messages=msgs,
                               temperature=0.0, max_tokens=512)
        raw = r["choices"][0]["message"]["content"]
    except Exception:
        return ""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL).strip()
    m = re.search(r'Search Query:\s*"?(.+?)"?\s*$', raw, re.I | re.M)
    return m.group(1).strip().strip('"\'') if m else ""


# ── Main loop ─────────────────────────────────────────────────────

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

    # Round 1: Planner decomposes the question
    all_queries = plan(client, model, query)
    if not all_queries:
        all_queries = [query]
    empty_streak = 0

    for rnd in range(1, max_turns + 1):
        rec: Dict[str, Any] = {"queries": all_queries}
        all_docs: List[Dict] = []
        all_results: List[Dict] = []

        # ── Search + Rerank + Fetch per query ──
        for q in all_queries:
            if not q or not q.strip():
                continue
            q = q.strip()
            if qh.was_recent(q):
                continue
            qh.add(q)

            # Two-stage retrieval
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

            # Fetch top-3 full documents (to avoid blowing context)
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
                        "docid": did,
                        "text": doc.get("text", ""),
                        "url": doc.get("url", ""),
                    })

        rec["results"] = all_results
        rec["fetched"] = all_docs

        if not all_docs:
            empty_streak += 1
            # Try rethink or replan
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

        # ── Execute: extract structured evidence ──
        search_q = all_queries[0] if all_queries else query
        evidence_list = execute(client, model, query, all_docs, search_q)
        added = evidence.add_batch(evidence_list, query=search_q, round_num=rnd)
        rec["extract"] = json.dumps(evidence_list, ensure_ascii=False)

        if added == 0:
            empty_streak += 1
        else:
            empty_streak = 0

        # ── Assess: gap analysis ──
        assess_result = assess(client, model, query, evidence, qh)
        rec["assess"] = (
            f"Status: {assess_result['status']}\n"
            + (f"Known: {'; '.join(assess_result['known_facts'][:5])}\n" if assess_result['known_facts'] else "")
            + (f"Missing: {'; '.join(assess_result['missing'][:5])}\n" if assess_result['missing'] else "")
            + (f"Next Queries: {', '.join(assess_result['next_queries'][:5])}" if assess_result['next_queries'] else "")
        )
        round_records.append(rec)

        if assess_result["status"] == "READY_TO_ANSWER":
            break

        # ── Next queries ──
        next_queries = assess_result.get("next_queries", [])
        if next_queries:
            # Filter out recent duplicates
            filtered = [q for q in next_queries if not qh.was_recent(q)]
            if filtered:
                all_queries = filtered
                continue

        # Stuck: use rethink or replan
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

        # Final fallback: replan with context
        ctx = f"Findings so far: {evidence.summary_for_context(max_items=5)}"
        new_queries = plan(client, model, query, ctx)
        if new_queries:
            all_queries = [q for q in new_queries if not qh.was_recent(q)]
            if all_queries:
                continue

        break

    # ── Synthesize final answer ──
    final_answer = synthesize(client, model, query, evidence)
    if not final_answer or final_answer.startswith("ERROR"):
        final_answer = "Unable to determine answer from available evidence."

    trajectory = build_trajectory(query, evidence, qh, round_records, final_answer)
    return final_answer, trajectory
