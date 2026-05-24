"""Planner agent — decompose complex questions into BM25-friendly keyword queries."""

import re
from typing import List

PLANNER_SYSTEM = (
    "Break complex questions into searchable sub-queries. "
    "Output one per line with '- ' prefix."
)

PLANNER_PROMPT = """Decompose this question into 3-5 search queries. Each query must include specific entities, dates, and details from the question. Output ONLY lines starting with '- '.

Example:
Question: A restaurant founded in the 1950s in Chicago by a chef trained in France in the 1940s, with a signature dish technique on pages 50-55 of their cookbook. The chef's mentor worked at a Paris hotel in the 1920s. Name the first executive pastry chef.

Output:
- 1950s Chicago restaurant French-trained chef 1940s
- signature dish cooking technique pages 50-55 cookbook
- chef mentor Paris hotel 1920s
- first executive pastry chef restaurant name"""


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


def plan(client, model, question: str,
         extra_context: str = "") -> List[str]:
    user = f"## Question\n{question}\n\n{PLANNER_PROMPT}"
    if extra_context:
        user = (
            f"## Question\n{question}\n\n"
            f"## What We Already Know\n{extra_context}\n\n"
            f"{PLANNER_PROMPT}\nFocus on what is still MISSING."
        )
    msgs = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = _strip(_chat(client, model, msgs, max_tok=512))
    queries = _parse_queries(raw)
    return queries if queries else [question]
