# -*- coding: utf-8 -*-
"""
ReAct Agent — 纯净版，无状态反馈、无强制提示。
"""

import json, re
from typing import Any, Dict, List, Tuple
from enum import Enum


class State(Enum):
    THINK = "think"
    OBSERVE = "observe"
    ANSWER = "answer"
    DONE = "done"


class EvidenceStore:
    """累积证据，替换旧原始数据为摘要。"""

    def __init__(self, question: str):
        self.question = question
        self.facts: List[str] = []
        self.doc_finds: Dict[str, str] = {}
        self.searched: List[str] = []
        self._fact_set: set = set()

    def add_search_result(self, query: str, result_json: str):
        self.searched.append(query)
        try:
            items = json.loads(result_json)
            if not isinstance(items, list): return
            for item in items[:3]:
                docid = item.get("docid", "")
                snippet = item.get("snippet", "")
                if docid and snippet:
                    q_words = set(re.findall(r'[a-z]{3,}', self.question.lower()))
                    s_words = set(re.findall(r'[a-z]{3,}', snippet.lower()))
                    if len(q_words & s_words) >= 1 or len(snippet) > 50:
                        self.doc_finds[docid] = snippet[:400]
        except (json.JSONDecodeError, TypeError): pass

    def add_document(self, docid: str, result_json: str):
        try:
            doc = json.loads(result_json)
            text = doc.get("text_preview", "") or doc.get("text", "")
            if not text: return
            sentences = re.split(r'[.!?\n]+', text)
            q_words = set(re.findall(r'[a-z]{3,}', self.question.lower()))
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20 or len(sent) > 300: continue
                s_words = set(re.findall(r'[a-z]{3,}', sent.lower()))
                overlap = q_words & s_words
                has_entity = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', sent))
                has_year = bool(re.search(r'\b(?:1[6-9]\d{2}|20[0-2]\d)\b', sent))
                if len(overlap) >= 2 or (len(overlap) >= 1 and (has_entity or has_year)):
                    k = sent[:80].lower().strip()
                    if k not in self._fact_set:
                        self._fact_set.add(k)
                        self.facts.append(f"[doc {docid}] {sent[:200]}")
            self.doc_finds[docid] = text[:300]
        except (json.JSONDecodeError, TypeError): pass

    def summary(self) -> str:
        parts = []
        if self.facts:
            parts.append("## 已发现的关键事实")
            for f in self.facts[-15:]: parts.append(f"- {f}")
        if self.doc_finds:
            parts.append("\n## 已检索的文档")
            for docid, snippet in list(self.doc_finds.items())[-8:]:
                parts.append(f"- docid={docid}: {snippet[:120]}")
        if self.searched:
            parts.append("\n## 已搜索的查询")
            for q in self.searched[-8:]: parts.append(f"- {q}")
        return "\n".join(parts) if parts else "(暂无已知信息)"

    def is_empty(self) -> bool: return not self.facts and not self.doc_finds


SYSTEM_PROMPT = """你是一个深度研究智能体。通过搜索文档语料库来回答复杂问题。

## 可用工具
- `search(query)` — BM25 关键词搜索，返回 top-5 结果。查询词: 2-5 个英文实词。
- `get_document(docid)` — 根据 docid 获取文档全文。

## 方法: Thought → Action → Observation 循环
1. 从问题提取关键词 search
2. 看到相关线索 → get_document 读全文
3. 发现缺口 → 用不同关键词再 search
4. 证据够了 → 给出最终答案

## 停止条件
收集到足够事实时输出最终答案。

## 最终答案格式
Explanation: <推理过程>
Exact Answer: <精确答案>"""


def _execute_tool_call(tool_call: Dict[str, Any], registry: Dict[str, Any]) -> str:
    fn_name = tool_call["function"]["name"]
    try: args = json.loads(tool_call["function"]["arguments"])
    except (json.JSONDecodeError, KeyError): return json.dumps({"error": "参数解析失败"})
    fn = registry.get(fn_name)
    if fn is None: return json.dumps({"error": f"未知工具: {fn_name}"})
    try: result = fn(**args)
    except Exception as e: return json.dumps({"error": str(e)})
    result_str = json.dumps(result, ensure_ascii=False)
    limit = 8000
    if len(result_str) > limit: result_str = result_str[:limit] + "... [truncated]"
    return result_str


def _extract_final_answer(text: str) -> str:
    m = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.I)
    if m: return m.group(1).strip()
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.I)
    if m: return m.group(1).strip()
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if lines: return lines[-1]
    return text.strip()


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _build_messages(system_prompt: str, question: str, evidence: EvidenceStore,
                    recent_messages: List[Dict]) -> List[Dict]:
    evidence_text = evidence.summary()
    system_content = system_prompt
    if not evidence.is_empty():
        system_content += f"\n\n## 当前已知信息\n{evidence_text}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
    messages.extend(recent_messages)
    return messages


def run_agent_loop(
    client, model: str, query: str,
    tools: List[Dict[str, Any]], registry: Dict[str, Any],
    max_turns: int = 10, max_history_msgs: int = 6,
    min_searches: int = 2,
) -> Tuple[str, List[Dict[str, Any]]]:

    trajectory: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    evidence = EvidenceStore(query)
    state = State.THINK
    turn = 0
    total_tool_calls = 0
    max_total_calls = max_turns * 3
    recent_raw: List[Dict] = []
    max_recent = max_history_msgs
    max_recent_chars = 30000
    search_count = 0
    doc_count = 0
    last_queries: List[str] = []

    while state != State.DONE:
        if state == State.THINK:
            turn += 1
            if turn > max_turns:
                state = State.ANSWER
                continue

            model_messages = _build_messages(
                system_prompt=SYSTEM_PROMPT, question=query,
                evidence=evidence, recent_messages=recent_raw[-max_recent:],
            )

            try:
                response = client.simple_chat(
                    model=model, messages=model_messages, tools=tools,
                    temperature=0.0, max_tokens=4096,
                )
            except Exception as e:
                trajectory.append({"role": "assistant", "content": f"ERROR: {e}"})
                state = State.DONE; continue

            assistant_msg = response["choices"][0]["message"]
            msg_to_append: Dict[str, Any] = {"role": "assistant"}
            if assistant_msg.get("content"):
                msg_to_append["content"] = assistant_msg["content"]
            if assistant_msg.get("tool_calls"):
                msg_to_append["tool_calls"] = assistant_msg["tool_calls"]

            trajectory.append(msg_to_append)
            recent_raw.append(msg_to_append)

            tool_calls = assistant_msg.get("tool_calls") or []
            if tool_calls:
                state = State.OBSERVE
            else:
                state = State.ANSWER

        elif state == State.OBSERVE:
            last_assistant = trajectory[-1]
            tool_calls = last_assistant.get("tool_calls", [])

            for tc in tool_calls:
                total_tool_calls += 1
                if total_tool_calls > max_total_calls:
                    tool_msg = {"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"error": "已达最大工具调用次数"})}
                    trajectory.append(tool_msg); recent_raw.append(tool_msg); continue

                fn_name = tc["function"]["name"]
                # 去重
                if fn_name == "search":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        current_q = args.get("query", "").strip().lower()
                        if current_q and current_q in last_queries:
                            tool_msg = {"role": "tool", "tool_call_id": tc["id"],
                                "content": json.dumps([{
                                    "docid":"","score":0,
                                    "snippet":f"已搜索过 '{current_q}'，换不同关键词"}])}
                            trajectory.append(tool_msg); recent_raw.append(tool_msg); continue
                        if current_q: last_queries.append(current_q)
                    except (json.JSONDecodeError, KeyError): pass

                result = _execute_tool_call(tc, registry)
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result}
                trajectory.append(tool_msg); recent_raw.append(tool_msg)

                if fn_name == "search":
                    search_count += 1
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        evidence.add_search_result(args.get("query", ""), result)
                    except (json.JSONDecodeError, KeyError): pass
                elif fn_name == "get_document":
                    doc_count += 1
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        evidence.add_document(args.get("docid", ""), result)
                    except (json.JSONDecodeError, KeyError): pass

            # 裁剪
            total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in recent_raw)
            while total_chars > max_recent_chars and len(recent_raw) > 2:
                removed = recent_raw.pop(0)
                total_chars -= len(json.dumps(removed, ensure_ascii=False))

            state = State.THINK

        elif state == State.ANSWER:
            final_answer = ""
            for msg in reversed(trajectory):
                if msg.get("role") == "assistant" and msg.get("content"):
                    cleaned = _strip_thinking(msg["content"])
                    if cleaned:
                        final_answer = _extract_final_answer(cleaned)
                    break
            if not final_answer:
                final_answer = "Unable to determine from available evidence."

            state = State.DONE

    return final_answer, trajectory
