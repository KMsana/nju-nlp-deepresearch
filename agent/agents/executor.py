"""Executor agent — extract structured JSON evidence from documents."""

import json
import re
from typing import Dict, List

from ..retrieval.chunking import QueryAwareChunker

EXECUTOR_SYSTEM = (
    "You are a precise evidence extraction agent. "
    "Extract specific, verifiable facts from documents as structured JSON. "
    "Only include facts directly stated in the documents. Do not infer."
)

EXECUTOR_PROMPT = """## Question
{question}

## Documents
{documents}

## Task
Extract ALL facts relevant to answering the question. For each fact, include the supporting quote.

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


def _parse_json_output(raw: str) -> List[Dict]:
    cleaned = _strip(raw)
    # Try code fence extraction
    fence = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    # Find outermost JSON array
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
            query if query else "", text, max_total_chars=5000,
        )
        parts.append(
            f"--- Document {i} (docid={d['docid']}, "
            f"url={d.get('url', '')}) ---\n{chunked}"
        )
    return "\n\n".join(parts) if parts else "(no documents)"


def execute(client, model, question: str,
            docs: List[Dict], search_query: str = "") -> List[Dict]:
    if not docs:
        return []
    msgs = [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {"role": "user", "content": EXECUTOR_PROMPT.format(
            question=question,
            documents=_fmt_docs(docs, search_query),
        )},
    ]
    raw = _chat(client, model, msgs, max_tok=2048)
    return _parse_json_output(raw)
