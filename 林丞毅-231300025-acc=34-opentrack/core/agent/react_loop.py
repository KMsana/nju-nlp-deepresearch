# -*- coding: utf-8 -*-
"""
ReAct 状态机 Agent — Thought → Action → Observation 循环。

状态:
  THINK  → 模型推理，决定调工具还是直接回答
  OBSERVE → 代码执行工具，返回结果
  ANSWER → 模型给出最终答案
  DONE   → 结束

上下文管理：EvidenceStore 累积证据，替换旧原始数据为精炼摘要。

18准确率
"""

import json
import re
from typing import Any, Dict, List, Tuple
from enum import Enum


# ── 状态定义 ─────────────────────────────────────────────────────────

class State(Enum):
    THINK = "think"
    OBSERVE = "observe"
    ANSWER = "answer"
    DONE = "done"


# ── 证据仓库 ─────────────────────────────────────────────────────────

class EvidenceStore:
    """累积与问题相关的证据，用于替换旧的原始工具返回。"""

    def __init__(self, question: str):
        self.question = question
        self.facts: List[str] = []           # 提取的关键事实
        self.doc_finds: Dict[str, str] = {}  # docid → 关键内容摘要
        self.searched: List[str] = []        # 已搜索的查询词
        self._fact_set: set = set()          # 去重用

    def add_search_result(self, query: str, result_json: str):
        """从搜索结果中提取有价值的 docid 和线索。"""
        self.searched.append(query)
        try:
            items = json.loads(result_json)
            if not isinstance(items, list):
                return
            for item in items[:3]:
                docid = item.get("docid", "")
                snippet = item.get("snippet", "")
                if docid and snippet:
                    # 只保留和问题有关键词重叠的
                    q_words = set(re.findall(r'[a-z]{3,}', self.question.lower()))
                    s_words = set(re.findall(r'[a-z]{3,}', snippet.lower()))
                    overlap = len(q_words & s_words)
                    if overlap >= 1 or len(snippet) > 50:
                        self.doc_finds[docid] = snippet[:400]
        except (json.JSONDecodeError, TypeError):
            pass

    def add_document(self, docid: str, result_json: str):
        """从文档全文中提取关键事实。"""
        try:
            doc = json.loads(result_json)
            text = doc.get("text_preview", "") or doc.get("text", "")
            if not text:
                return
            # 提取事实：包含名字、日期、书名等具体信息的句子
            self._extract_facts_from_text(text, docid)
            self.doc_finds[docid] = text[:300]
        except (json.JSONDecodeError, TypeError):
            pass

    def _extract_facts_from_text(self, text: str, docid: str):
        """启发式提取关键事实：包含实体名、年份、书名的句子。"""
        # 按句分割
        sentences = re.split(r'[.!?\n]+', text)
        q_words = set(re.findall(r'[a-z]{3,}', self.question.lower()))

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue

            # 计算与问题的关键词重叠
            s_words = set(re.findall(r'[a-z]{3,}', sent.lower()))
            overlap = q_words & s_words

            # 有价值的句子：和问题有重叠，或包含具体实体
            has_entity = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', sent))
            has_year = bool(re.search(r'\b(?:1[6-9]\d{2}|20[0-2]\d)\b', sent))

            if len(overlap) >= 2 or (len(overlap) >= 1 and (has_entity or has_year)):
                fact_key = sent[:80].lower().strip()
                if fact_key not in self._fact_set:
                    self._fact_set.add(fact_key)
                    self.facts.append(f"[doc {docid}] {sent[:200]}")

    def summary(self) -> str:
        """生成精炼的证据摘要，供模型参考。"""
        parts = []

        if self.facts:
            parts.append("## 已发现的关键事实")
            for f in self.facts[-15:]:  # 最多保留 15 条
                parts.append(f"- {f}")

        if self.doc_finds:
            parts.append("\n## 已检索的文档")
            for docid, snippet in list(self.doc_finds.items())[-8:]:  # 最多 8 个
                parts.append(f"- docid={docid}: {snippet[:120]}")

        if self.searched:
            parts.append("\n## 已搜索的查询")
            for q in self.searched[-8:]:
                parts.append(f"- {q}")

        return "\n".join(parts) if parts else "(暂无已知信息)"

    def is_empty(self) -> bool:
        return not self.facts and not self.doc_finds


# ── 系统提示词 ────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
你是一个深度研究智能体。你的任务是通过搜索文档语料库来回答复杂问题。

## 可用工具
1. `search(query)` — BM25 关键词搜索，返回 top-5 结果（含 docid、评分、摘要）。
   - 查询词要求：2-5 个英文实词，不要用句子或虚词。
   - BM25 是纯关键词匹配——只有文档中**完全包含**的词才能匹配。
2. `get_document(docid)` — 根据 docid 获取文档全文。摘要线索不够时，用这个工具读全文。

## 核心方法：Thought → Action → Observation 循环

每一步你必须遵循以下循环：

**Thought（推理）**：分析当前状态——
- 我已经知道了什么？（参考"已发现的关键事实"）
- 我还缺什么信息？
- 下一步最该做什么？为什么？

**Action（行动）**：基于推理，选择一个工具调用——
- `search`：用新关键词搜索（不要重复"已搜索的查询"中的词）
- `get_document`：读取某个文档的全文（当摘要线索不够时）

**Observation（观察）**：分析工具返回的结果——
- 这些结果回答了我的哪个疑问？
- 有没有新的线索可以继续追？

## 搜索策略

1. 第一轮：从问题中提取最独特的关键词搜索。
2. 看到线索：搜索结果中有相关但不完整的线索 → 调用 `get_document` 读全文。
3. 发现缺口：读完后仍有未解答的部分 → 用**不同角度**的关键词再搜索。
4. 综合判断：收集到足够事实时，停止搜索，给出最终答案。

**关键：不要只搜不读。看到有希望的文档，必须 `get_document` 读全文！**
**关键：不要重复搜索。每次查询必须用不同的关键词组合。**

## 停止条件
- 已找到足够证据 → 给出最终答案。
- 已搜索 3 次以上且读过全文，仍无新发现 → 基于已有证据给出最佳答案。
- **至少搜索 3 次不同角度才能放弃。只搜 1 次就回答是不够的。**
- **每次搜索后，必须对最有希望的文档调用 get_document 读全文。搜索摘要永远不够。**

## 最终答案格式
**不要再调用工具**，直接输出：
Explanation: <简要推理过程>
Exact Answer: <最终答案，专有名词或人名记得保证完整>"""


# ── 辅助函数 ──────────────────────────────────────────────────────────

def _execute_tool_call(tool_call: Dict[str, Any], registry: Dict[str, Any]) -> str:
    """执行单个工具调用，返回 JSON 字符串。"""
    fn_name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except (json.JSONDecodeError, KeyError):
        return json.dumps({"error": "参数解析失败"})

    fn = registry.get(fn_name)
    if fn is None:
        return json.dumps({"error": f"未知工具: {fn_name}"})

    try:
        result = fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})

    result_str = json.dumps(result, ensure_ascii=False)
    limit = 8000 if fn_name == "get_document" else 8000
    if len(result_str) > limit:
        result_str = result_str[:limit] + "... [truncated]"
    return result_str


def _extract_final_answer(text: str) -> str:
    """从模型输出中提取最终答案。"""
    m = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
    if cleaned:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if lines:
            return lines[-1]
    inner = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if inner:
        return '\n'.join(s.strip() for s in inner if s.strip())
    return text.strip()


def _build_messages_for_model(
    system_prompt: str,
    question: str,
    evidence: EvidenceStore,
    recent_messages: List[Dict],
    hint: str = "",
) -> List[Dict]:
    """构建发送给模型的 messages，用证据摘要替换旧原始数据。

    结构：
    [system: 提示词 + 证据摘要]
    [user: 问题]
    [最近几轮的完整消息]
    [可选: 状态反馈 hint]
    """
    # 系统提示词 + 证据摘要
    evidence_text = evidence.summary()
    system_content = system_prompt
    if not evidence.is_empty():
        system_content += f"\n\n## 当前已知信息\n{evidence_text}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    # 追加最近的交互消息
    messages.extend(recent_messages)

    # 追加状态反馈
    if hint:
        messages.append({"role": "system", "content": f"[状态反馈] {hint}"})

    return messages


def _build_hint(search_count: int, doc_count: int, empty_streak: int,
                last_queries: List[str], turn: int) -> str:
    """根据状态机指标，生成反馈提示。"""
    hints = []
    if search_count >= 1 and doc_count == 0:
        hints.append("你搜索了但从未读取文档全文。搜索结果只是摘要，信息不够。请立即对最相关的文档调用 get_document 读全文。")
    if empty_streak >= 2:
        recent = ", ".join(last_queries[-3:]) if last_queries else "(无)"
        hints.append(f"连续 {empty_streak} 次搜索无新发现。已搜索: {recent}。请换完全不同的关键词角度——想想目标文档会用什么词描述这个信息？")
    if turn >= 2 and search_count < 3:
        hints.append(f"你目前只搜索了 {search_count} 次。至少需要尝试 3 个不同角度的搜索才能放弃。换一组完全不同的关键词继续搜。")
    return "\n".join(hints)


# ── 状态机主循环 ──────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """移除 <think> 标签及其内容。"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def run_agent_loop(
    client,
    model: str,
    query: str,
    tools: List[Dict[str, Any]],
    registry: Dict[str, Any],
    max_turns: int = 5,
    max_history_msgs: int = 6,
    min_searches: int = 2,
) -> Tuple[str, List[Dict[str, Any]]]:
    """ReAct 状态机：Thought → Action → Observation 循环。

    上下文管理策略：
    - EvidenceStore 累积所有轮次的关键事实和文档发现
    - 每轮 THINK 时，用证据摘要 + 最近 N 条消息构建上下文
    - 旧的原始工具返回不发给模型，只保留精炼后的证据

    Returns (final_answer, trajectory_messages)。
    """
    # trajectory_messages: 完整轨迹（用于评估提交）
    trajectory: List[Dict[str, Any]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    evidence = EvidenceStore(query)
    state = State.THINK
    turn = 0
    total_tool_calls = 0
    max_total_calls = max_turns * 3

    # 最近几轮的原始消息（发给模型看的）
    recent_raw: List[Dict] = []
    max_recent = max_history_msgs  # 保留最近 N 条消息
    max_recent_chars = 30000       # recent_raw 总大小上限

    # 状态机指标
    search_count = 0
    doc_count = 0
    empty_streak = 0
    last_queries: List[str] = []

    while state != State.DONE:
        # ── THINK 状态 ──
        if state == State.THINK:
            turn += 1
            if turn > max_turns:
                state = State.ANSWER
                continue

            # 构建上下文：证据摘要 + 最近消息
            hint = _build_hint(search_count, doc_count, empty_streak, last_queries, turn)
            model_messages = _build_messages_for_model(
                system_prompt=REACT_SYSTEM_PROMPT,
                question=query,
                evidence=evidence,
                recent_messages=recent_raw[-max_recent:],
                hint=hint if turn > 1 else "",
            )

            try:
                response = client.simple_chat(
                    model=model,
                    messages=model_messages,
                    tools=tools,
                    temperature=0.0,
                    max_tokens=4096,
                )
            except Exception as e:
                trajectory.append({"role": "assistant", "content": f"ERROR: {e}"})
                state = State.DONE
                continue

            assistant_msg = response["choices"][0]["message"]

            # 构建消息
            msg_to_append: Dict[str, Any] = {"role": "assistant"}
            if assistant_msg.get("content"):
                msg_to_append["content"] = assistant_msg["content"]
            if assistant_msg.get("tool_calls"):
                msg_to_append["tool_calls"] = assistant_msg["tool_calls"]

            # 写入轨迹和最近消息
            trajectory.append(msg_to_append)
            recent_raw.append(msg_to_append)

            tool_calls = assistant_msg.get("tool_calls") or []

            if tool_calls:
                state = State.OBSERVE
            else:
                state = State.ANSWER

        # ── OBSERVE 状态 ──
        elif state == State.OBSERVE:
            last_assistant = trajectory[-1]
            tool_calls = last_assistant.get("tool_calls", [])

            for tc in tool_calls:
                total_tool_calls += 1
                if total_tool_calls > max_total_calls:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": "已达最大工具调用次数，请基于已有信息给出答案"}),
                    }
                    trajectory.append(tool_msg)
                    recent_raw.append(tool_msg)
                    continue

                fn_name = tc["function"]["name"]

                # 重复查询检测
                if fn_name == "search":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        current_q = args.get("query", "").strip().lower()
                        if current_q and current_q in last_queries:
                            dup_msg = {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps([{
                                    "docid": "", "score": 0,
                                    "snippet": f"你已经搜索过 '{current_q}'，请换一组完全不同的关键词。",
                                }]),
                            }
                            trajectory.append(dup_msg)
                            recent_raw.append(dup_msg)
                            empty_streak += 1
                            continue
                        if current_q:
                            last_queries.append(current_q)
                    except (json.JSONDecodeError, KeyError):
                        pass

                # 执行工具
                result = _execute_tool_call(tc, registry)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
                trajectory.append(tool_msg)
                recent_raw.append(tool_msg)

                # 更新证据仓库
                if fn_name == "search":
                    search_count += 1
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        evidence.add_search_result(args.get("query", ""), result)
                    except (json.JSONDecodeError, KeyError):
                        pass
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, list) and parsed and parsed[0].get("docid"):
                            empty_streak = 0
                        else:
                            empty_streak += 1
                    except (json.JSONDecodeError, TypeError):
                        empty_streak += 1
                elif fn_name == "get_document":
                    doc_count += 1
                    empty_streak = 0
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        evidence.add_document(args.get("docid", ""), result)
                    except (json.JSONDecodeError, KeyError):
                        pass

            # 搜索后强制提示读文档：搜到了但从未读全文时，列出可用 docid
            if doc_count == 0 and search_count >= 1:
                found_docids = list(evidence.doc_finds.keys())[-5:]
                if found_docids:
                    force_hint = {
                        "role": "system",
                        "content": (
                            f"[状态反馈] 你已搜索 {search_count} 次但从未读取文档全文。"
                            f"搜索摘要信息不足，必须读全文。请对以下某个 docid 调用 get_document：{found_docids}"
                        ),
                    }
                    trajectory.append(force_hint)
                    recent_raw.append(force_hint)

            # 裁剪 recent_raw，保持总大小在预算内
            total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in recent_raw)
            while total_chars > max_recent_chars and len(recent_raw) > 2:
                removed = recent_raw.pop(0)
                total_chars -= len(json.dumps(removed, ensure_ascii=False))

            state = State.THINK

        # ── ANSWER 状态 ──
        elif state == State.ANSWER:
            # 最低搜索次数检查：还没搜够就强制回去继续搜
            if search_count < min_searches and turn <= max_turns:
                hint_msg = {
                    "role": "system",
                    "content": f"[状态反馈] 你只搜索了 {search_count} 次（至少需要 {min_searches} 次）。请继续搜索——用不同的关键词角度。",
                }
                trajectory.append(hint_msg)
                recent_raw.append(hint_msg)
                state = State.THINK
                continue

            # 提取最终答案，确保不含 think 标签
            final_answer = ""
            for msg in reversed(trajectory):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    # 清理 think 标签
                    cleaned = _strip_thinking(content)
                    if cleaned:
                        final_answer = _extract_final_answer(cleaned)
                    break

            if not final_answer:
                final_answer = "Unable to determine from available evidence."

            state = State.DONE

    return final_answer, trajectory
