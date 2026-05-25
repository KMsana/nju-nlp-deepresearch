import re
from typing import Any, Callable, Dict, List, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    def find_in_doc(docid: str, keyword: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        text = doc.get("text", "")
        if not text:
            return {"docid": docid, "error": "document is empty"}
        matches = [m.start() for m in re.finditer(re.escape(keyword), text, re.IGNORECASE)]
        if not matches:
            return {"docid": docid, "keyword": keyword, "message": "keyword not found"}
        snippets = []
        for idx in matches[:5]:
            start = max(0, idx - 100)
            end = min(len(text), idx + len(keyword) + 100)
            snippets.append(text[start:end].replace('\n', ' '))
        return {"docid": docid, "keyword": keyword, "match_count": len(matches), "snippets": snippets}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by its docid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_doc",
                "description": "Find a keyword within a document and return surrounding context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "keyword": {"type": "string", "description": "Keyword to search within the document"},
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
    ]
    return tools, {"search": search, "get_document": get_document, "find_in_doc": find_in_doc, "_searcher": searcher}


# ══════════════════════════════════════════════════════════════════════
# LLM-powered tools (call the model internally)
# ══════════════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """
将问题分解为3-5个英文关键词搜索方向。每个方向覆盖不同角度。

输出每行一个方向：
- keyword1 keyword2 keyword3
- keyword4 keyword5 keyword6"""

VERIFY_PROMPT = """
逐条检查候选答案的每个声明是否在事实依据中有文档明确支撑。

全部有支撑 → Verdict: VERIFIED
有声明缺支撑 → Verdict: UNVERIFIED
Missing: <缺少支撑的内容>

输出：
Verdict: VERIFIED | UNVERIFIED
Missing: ..."""


def llm_decompose(client, model, question: str) -> list:
    """LLM 工具：将复杂问题分解为关键词搜索方向。"""
    msgs = [
        {"role": "system", "content": "将复杂问题分解为BM25关键词搜索方向。"},
        {"role": "user", "content": f"## 问题\n{question}\n\n{DECOMPOSE_PROMPT}"},
    ]
    try:
        raw = client.simple_chat(model=model, messages=msgs, temperature=0.0, max_tokens=512)
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        return []
    angles = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('-'):
            q = s[1:].strip().strip('"\'')
            q = q.replace('**', '').replace('*', '').strip()
            if len(q.split()) >= 2:
                angles.append(q)
    return angles[:5]


def llm_verify(client, model, question: str, candidate_answer: str,
               facts_text: str) -> dict:
    """LLM 工具：验证候选答案是否有事实支撑。"""
    ctx = (f"## 问题\n{question}\n\n"
           f"## 候选答案\n{candidate_answer}\n\n"
           f"## 事实依据\n{facts_text}\n\n"
           f"{VERIFY_PROMPT}")
    msgs = [
        {"role": "system", "content": "验证答案的每个声明是否在事实中有文档支撑。"},
        {"role": "user", "content": ctx},
    ]
    try:
        raw = client.simple_chat(model=model, messages=msgs, temperature=0.0, max_tokens=512)
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        return {"verdict": "UNVERIFIED", "missing": "LLM call failed"}
    verdict = "VERIFIED" if re.search(r'Verdict:\s*VERIFIED', text, re.I) else "UNVERIFIED"
    missing = ""
    m = re.search(r'Missing:\s*(.+?)(?:\n|$)', text, re.I | re.DOTALL)
    if m:
        missing = m.group(1).strip()
    return {"verdict": verdict, "missing": missing}
