# -*- coding: utf-8 -*-
"""OpenTrack 独立运行入口。从 open_track/ 目录执行。"""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.agent.vllm_client import VLLMClient
from core.agent.tools import build_searcher, get_agent_tool_specs_and_registry
from core.agent.dataset_utils import load_jsonl
from agent.multiagent import run_agent_loop

# ── 配置 ──
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "qwen_auto"
API_KEY = "dummy"
INDEX_PATH = "indexes/browsecomp_plus_bm25.sqlite"
DATASET_PATH = "browsecomp_plus_hard50.jsonl"
OUTPUT_PATH = "runs/submission.jsonl"
MAX_TURNS = 5
WORKERS = 3


def process_one(row):
    client = VLLMClient(base_url=VLLM_BASE_URL, api_key=API_KEY)
    searcher = build_searcher(index_path=INDEX_PATH)
    tools, registry = get_agent_tool_specs_and_registry(searcher=searcher)

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


def main():
    rows = load_jsonl(DATASET_PATH)
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
                ans = rec["predicted_answer"][:60].replace("\n", " ")
                print(f"[{i+1}/{len(rows)}] {rec['query_id']} -> {ans}", flush=True)
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
