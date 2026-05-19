import json
import copy
from typing import Any, Dict, List, Tuple

system_prompt_template = """你是一个智能的AI深度研究助手。

【当前目标和已知信息】
你的目标是通过多轮检索，从本地知识库中收集确实的证据来准确回答用户的问题。在检索之前，你没有任何关于该问题的内建已知事实，必须全部依赖检索结果。

【可用工具】
- search(query): 搜索知识库，返回相关文档的 docid 及相关内容片段（snippet）。如果首次搜索证据不足，请尝试改写 query 继续搜索。
- get_document(docid): 根据搜到的 docid，获取并阅读该文档的完整正文内容。
- find_in_doc(docid, keyword): 在指定的某篇长文档中搜索特定字段(keyword)，快速返回上下文字段，避免读取过长文档。

【每步输出的格式】
1. 在决定调用工具之前，请在内容(content)中先写下你的分析思考过程（Thought），如“我看了XX文档，但这还不够，我还需要查YY...”。
2. 当你需要输出最终答案（停止搜索）时，请严格使用如下固定格式作答：
Explanation: <简明扼要的证据总结与推导过程>
Exact Answer: <最终的简短、精确的答案>

【什么情况下停止搜索】
请在满足以下任一条件时停止调用工具，直接输出上述格式的最终回答：
1. 已找到可信度高、能直接回答该问题的答案。
2. 检索获得的信息已经在重复，或者你判断即使再改写 query 也查不到新信息。
"""

def truncate_observation(text: str, max_chars: int = 1500) -> str:
    """如果单次工具的返回过长，进行中间截断，保留头尾。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n...[内容过长，因长度限制被截断]...\n\n" + text[-half:]

def run_agent_loop(
    client, 
    model: str, 
    query: str, 
    tools: List[Dict[str, Any]], 
    registry: Dict[str, Any], 
    max_turns: int = 5,
    max_history_msgs: int = 6
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    运行 Deep Research Agent 主循环。
    :param max_history_msgs: 滑动窗口最大保留消息条数。
    """
    
    system_msg = {"role": "system", "content": system_prompt_template}
    initial_user_msg = {"role": "user", "content": query}
    
    full_history = [system_msg, initial_user_msg]
    rolling_history = []
    
    answer_text = ""
    
    for turn in range(max_turns):
        # 组装滑动窗口消息
        messages = [system_msg, initial_user_msg] + rolling_history
        
        # 调用大模型
        response = client.simple_chat(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            tools=tools
        )
        
        choice = response["choices"][0]
        response_msg = choice["message"]
        
        full_history.append(response_msg)
        rolling_history.append(response_msg)
        
        tool_calls = response_msg.get("tool_calls", [])
        if not tool_calls:
            # 模型认为不需要继续搜索，输出了自然文本回答，终止循环
            answer_text = response_msg.get("content", "")
            break
            
        # 动态分发工具解析
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]
            try:
                arguments = json.loads(arguments_str)
            except Exception:
                arguments = {}
                
            tool_result = ""
            if function_name in registry:
                try:
                    # 运行工具
                    result_obj = registry[function_name](**arguments)
                    # 格式化并截断长文本
                    tool_result = truncate_observation(json.dumps(result_obj, ensure_ascii=False, indent=2))
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"
            else:
                tool_result = f"Error: Tool {function_name} not found."
                
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": str(tool_result)
            }
            full_history.append(tool_msg)
            rolling_history.append(tool_msg)
            
        # 滑动窗口截断：防止超长
        if len(rolling_history) > max_history_msgs:
            rolling_history = rolling_history[-max_history_msgs:]
            
    # 如果超时
    if answer_text == "":
        force_stop_sys_msg = {
            "role": "system", 
            "content": "你已用尽最大允许的检索次数，请立即根据现有的观察结果给出你最好的答案，不要再使用工具。"
        }
        messages = [force_stop_sys_msg, initial_user_msg] + rolling_history
        response = client.simple_chat(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )
        final_msg = response["choices"][0]["message"]
        full_history.append(final_msg)
        answer_text = final_msg.get("content", "")
        
    return answer_text, full_history
