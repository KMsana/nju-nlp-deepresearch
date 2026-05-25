# -*- coding: utf-8 -*-
"""Multi-threaded submission generator — each worker owns its own client/searcher."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "qwen_auto"
API_KEY = "dummy"
INDEX_PATH = "indexes/browsecomp_plus_bm25.sqlite"
DATASET_PATH = "browsecomp_plus_hard50.jsonl"
OUTPUT_PATH = "runs/submission.jsonl"
MAX_TURNS = 5
WORKERS = 3
LIMIT = None  # None = all, or set to N for first N questions

# ── Per-worker init ──────────────────────────────────────────────────

def _build_worker():
    """Each worker thread creates its OWN client and searcher — no shared state."""
    from agent.vllm_client import VLLMClient
    from agent.tools import build_searcher, get_agent_tool_specs_and_registry

    client = VLLMClient(base_url=VLLM_BASE_URL, api_key=API_KEY)
    searcher = build_searcher(index_path=INDEX_PATH)
    tools, registry = get_agent_tool_specs_and_registry(searcher=searcher)
    return client, tools, registry


# ── Single question processor ────────────────────────────────────────

def process_one(row):
    from agent.deepresearch_agent import run_agent_loop

    client, tools, registry = _build_worker()
    answer, history = run_agent_loop(
        client=client, model=MODEL_NAME, query=row["query"],
        tools=tools, registry=registry, max_turns=MAX_TURNS,
    )
    return {
        "query_id": row["query_id"],
        "status": "completed",
        "predicted_answer": answer,
        "messages": history,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    from agent.dataset_utils import load_jsonl

    rows = load_jsonl(DATASET_PATH, limit=LIMIT)
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, row): i for i, row in enumerate(rows)}
        for f in as_completed(futures):
            i = futures[f]
            try:
                rec = f.result()
                results[i] = rec
                ans_preview = rec["predicted_answer"][:60].replace("\n", " ")
                print(f"[{i+1}/{len(rows)}] {rec['query_id']} -> {ans_preview}", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(rows)}] FAILED: {e}", flush=True)
                results[i] = {
                    "query_id": rows[i]["query_id"],
                    "status": "error",
                    "predicted_answer": f"ERROR: {e}",
                    "messages": [],
                }

    with out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} records -> {out}")


if __name__ == "__main__":
    main()
