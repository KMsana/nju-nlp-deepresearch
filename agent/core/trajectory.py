"""Build real tool-call trajectories — actual snippets and text previews, not skeletons."""

import json
from typing import Any, Dict, List

from ..memory.evidence_store import EvidenceStore
from ..memory.query_history import QueryHistory


def build_trajectory(
    question: str,
    evidence_store: EvidenceStore,
    query_history: QueryHistory,
    round_records: List[Dict],
    final_answer: str,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Deep research agent — structured evidence pipeline."},
        {"role": "user", "content": question},
    ]
    call_id = 0

    for rec in round_records:
        # ── Search ──
        for q in rec.get("queries", []):
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{call_id}", "type": "function",
                    "function": {"name": "search",
                                 "arguments": json.dumps({"query": q},
                                                         ensure_ascii=False)},
                }],
            })
            results = rec.get("results", [])
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps([
                    {"docid": r["docid"], "score": r.get("score", 0),
                     "snippet": (r.get("snippet", r.get("text", "")) or "")[:300]}
                    for r in results
                ], ensure_ascii=False),
            })

        # ── Get document ──
        fetched = rec.get("fetched", [])
        if fetched:
            call_id += 1
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{call_id}", "type": "function",
                    "function": {"name": "get_document",
                                 "arguments": json.dumps({"docid": d["docid"]},
                                                         ensure_ascii=False)},
                } for d in fetched],
            })
            messages.append({
                "role": "tool", "tool_call_id": f"call_{call_id}",
                "content": json.dumps([
                    {"docid": d["docid"],
                     "text_preview": (d.get("text", "") or "")[:500],
                     "url": d.get("url", "")}
                    for d in fetched
                ], ensure_ascii=False),
            })

        # ── Extract ──
        if rec.get("extract"):
            messages.append({"role": "assistant",
                             "content": rec["extract"]})

        # ── Assess ──
        if rec.get("assess"):
            messages.append({"role": "assistant",
                             "content": rec["assess"]})

    # ── Final answer ──
    messages.append({"role": "assistant", "content": final_answer})
    return messages
