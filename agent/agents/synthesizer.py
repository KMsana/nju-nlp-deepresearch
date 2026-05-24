"""Synthesizer agent — produce final answer from structured evidence."""

import re
from typing import Dict

from ..memory.evidence_store import EvidenceStore

SYNTHESIZER_SYSTEM = (
    "Answer the question based strictly on the collected evidence. "
    "Do not fabricate information."
)

SYNTHESIZER_PROMPT = """## Question
{question}

{evidence_summary}

## Task
Based on the evidence above, what is the answer? Be precise.

Output exactly:
Exact Answer: <answer>

If evidence is insufficient, output:
Exact Answer: Unable to determine from available evidence."""


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


def synthesize(client, model, question: str,
               evidence_store: EvidenceStore) -> str:
    ctx = SYNTHESIZER_PROMPT.format(
        question=question,
        evidence_summary=evidence_store.full_summary(),
    )
    msgs = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {"role": "user", "content": ctx},
    ]
    raw = _strip(_chat(client, model, msgs, max_tok=512))

    m = re.search(r'Exact Answer:\s*(.+)', raw, re.I)
    if m:
        return m.group(1).strip()

    # Fallback: take the first meaningful line
    for line in raw.split('\n'):
        s = line.strip()
        if s and len(s) > 5 and not s.startswith(('Exact Answer:', 'ERROR:')):
            return s
    return raw.strip() or "Unable to determine answer from available evidence."
