"""Assessor agent — gap analysis, constraint audit, next-query generation."""

import re
from typing import Dict, List

from ..memory.evidence_store import EvidenceStore
from ..memory.query_history import QueryHistory

ASSESS_SYSTEM = (
    "You audit research progress. "
    "Check what is known, what is missing, and suggest next queries."
)

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


def _chat(client, model, msgs, max_tok=1024):
    try:
        r = client.simple_chat(model=model, messages=msgs,
                               temperature=0.0, max_tokens=max_tok)
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _strip(text):
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return re.sub(r'<think>.*', '', t, flags=re.DOTALL).strip() or text.strip()


def _parse_section(text: str, heading: str) -> List[str]:
    items = []
    in_block = False
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


def _parse_status(text: str) -> str:
    m = re.search(r'Status:\s*(READY_TO_ANSWER|NEED_MORE)', text, re.I)
    return m.group(1).upper() if m else "NEED_MORE"


def assess(client, model, question: str,
           evidence_store: EvidenceStore,
           query_history: QueryHistory) -> Dict:
    queries_text = "\n".join(
        f"  [{i+1}] {q}"
        for i, q in enumerate(query_history.recent_queries(8))
    ) or "(none)"

    ctx = ASSESS_PROMPT.format(
        question=question,
        evidence_summary=evidence_store.full_summary(),
        query_history=queries_text,
    )
    msgs = [
        {"role": "system", "content": ASSESS_SYSTEM},
        {"role": "user", "content": ctx},
    ]
    raw = _strip(_chat(client, model, msgs, max_tok=1024))

    status = _parse_status(raw)
    known = _parse_section(raw, "Known Facts")
    missing = _parse_section(raw, "Missing Information")
    next_queries = _parse_section(raw, "Next Queries")

    # Fallback: regex extract search queries
    if not next_queries and status == "NEED_MORE":
        for m in re.finditer(
            r'Search Query:\s*"?(.+?)"?\s*$', raw, re.I | re.M
        ):
            q = m.group(1).strip().strip('"\'')
            q = q.replace('**', '').replace('*', '').strip()
            if q:
                next_queries.append(q)

    return {
        "status": status,
        "known_facts": known,
        "missing": missing,
        "next_queries": next_queries,
        "raw": raw,
    }
