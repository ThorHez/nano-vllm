"""
OpenAI风格的ChatCompletion接口，支持工具执行

提供类似OpenAI Chat API的接口，支持工具调用和执行
"""

import json
import time
from typing import List, Dict, Any, Optional, Iterator, Union, Callable
from dataclasses import dataclass

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.tools.tool_executor import ToolExecutor
from nanovllm.tools.tool_definition import ToolDefinition
from nanovllm.tools.tool_call import ToolCall
from nanovllm.tools.tool_parser import ToolCallParser


@dataclass
class ChatMessage:
    """聊天消息"""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """聊天完成请求"""

    messages: List[Dict[str, Any]]
    model: str = "nanovllm"
    temperature: float = 0.8
    max_tokens: int = 200
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Union[str, Dict[str, Any]] = "auto"
    stream: bool = False


class ChatCompletionWithTools:
    """
    支持工具执行的ChatCompletion接口
    """

    def __init__(self, llm: LLM):
        """
        初始化ChatCompletion

        Args:
            llm: LLM实例
        """
        self.llm = llm
        self.tool_parser = ToolCallParser()
        self.tool_executor = ToolExecutor()

    def register_tool(self, tool: ToolDefinition):
        """注册工具"""
        self.tool_executor.register_tool(tool)

    def register_function_as_tool(
        self, func: Callable, name: str = None, description: str = None
    ):
        """将函数注册为工具"""
        self.tool_executor.register_function(func, name, description)

    def execute_tool_calls(
        self, tool_calls_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        执行工具调用

        Args:
            tool_calls_data: 工具调用数据列表，OpenAI格式

        Returns:
            执行结果列表
        """
        # 转换为ToolCall对象
        tool_calls = []
        for call_data in tool_calls_data:
            if call_data.get("type") == "function":
                func_data = call_data.get("function", {})
                name = func_data.get("name", "")
                arguments = func_data.get("arguments", "{}")

                # 解析JSON参数
                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=call_data.get("id", ""), name=name, arguments=arguments
                )
                tool_calls.append(tool_call)

        # 执行工具调用
        return self.tool_executor.execute_tool_calls(tool_calls)

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return self.tool_executor.get_available_tools()

    def create(
        self,
        messages: List[Dict[str, Any]],
        model: str = "nanovllm",
        temperature: float = 0.8,
        max_tokens: int = 200,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        stream: bool = False,
        execute_tools: bool = True,  # 是否自动执行工具调用
        **kwargs,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        创建聊天完成，支持工具执行

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大token数
            tools: 工具定义列表
            tool_choice: 工具选择策略
            stream: 是否流式返回
            execute_tools: 是否自动执行工具调用

        Returns:
            聊天完成响应或流式迭代器
        """
        # 构建提示词
        prompt = self._build_prompt(messages, tools, tool_choice)

        # 提取工具名称
        tool_names = None
        if tools:
            tool_names = [tool.get("function", {}).get("name") for tool in tools]
            tool_names = [name for name in tool_names if name]

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if stream:
            return self._create_stream(
                prompt, sampling_params, model, tool_names, execute_tools
            )
        else:
            return self._create_sync(
                prompt, sampling_params, model, tool_names, execute_tools
            )

    def _build_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
    ) -> str:
        """构建对话提示词"""
        prompt_parts = []

        # 添加工具定义信息
        if tools and tool_choice != "none":
            prompt_parts.append("你可以使用以下工具来帮助回答问题：\n")

            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    name = func.get("name", "")
                    description = func.get("description", "")
                    parameters = func.get("parameters", {})

                    tool_info = f"工具名称: {name}\n描述: {description}\n"

                    # 添加参数信息
                    if parameters and parameters.get("properties"):
                        tool_info += "参数:\n"
                        for param_name, param_info in parameters["properties"].items():
                            param_type = param_info.get("type", "string")
                            param_desc = param_info.get("description", "")
                            required = param_name in parameters.get("required", [])
                            req_str = "必需" if required else "可选"
                            tool_info += f"  - {param_name} ({param_type}, {req_str}): {param_desc}\n"
                    else:
                        tool_info += "参数: 无\n"

                    prompt_parts.append(tool_info)

            # 添加工具调用格式说明
            tool_usage = """
使用工具时，请按以下格式调用：
{"tool_call": {"name": "工具名称", "arguments": {"参数名": "参数值"}}}

或者使用函数调用格式：
工具名称(参数名="参数值")

请根据用户的问题选择合适的工具进行调用。
"""
            prompt_parts.append(tool_usage)

        # 添加对话历史
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
                # 如果有工具调用，也添加到提示词中
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        func = tool_call.get("function", {})
                        prompt_parts.append(
                            f"Tool call: {func.get('name')}({func.get('arguments', '')})"
                        )
            elif role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                prompt_parts.append(f"Tool result ({tool_call_id}): {content}")

        return "\n\n".join(prompt_parts) + "\n\nAssistant: "

    def _parse_tool_calls_from_text(
        self, text: str, tool_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        从文本中解析工具调用，返回OpenAI格式

        Args:
            text: 要解析的文本
            tool_names: 可用工具名称列表

        Returns:
            OpenAI格式的工具调用列表
        """
        if not tool_names or not text:
            return []

        try:
            # 使用工具执行器的解析器
            parsed_calls = self.tool_executor.parse_tool_calls(text)

            # 转换为OpenAI格式
            openai_calls = []
            for i, call in enumerate(parsed_calls):
                openai_calls.append(
                    {
                        "id": f"call_{int(time.time() * 1000)}_{i}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                )
            return openai_calls
        except Exception:
            return []

    def _create_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        model: str,
        tool_names: Optional[List[str]] = None,
        execute_tools: bool = True,
    ) -> Dict[str, Any]:
        """同步创建聊天完成"""
        start_time = time.time()

        # 生成响应
        results = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        result = results[0] if results else {"text": ""}

        generated_text = result["text"]

        # 从生成的文本中解析工具调用
        tool_calls = []
        tool_results = []

        if tool_names and generated_text:
            tool_calls = self._parse_tool_calls_from_text(generated_text, tool_names)

            # 如果需要执行工具调用
            if execute_tools and tool_calls:
                execution_results = self.execute_tool_calls(tool_calls)
                tool_results = execution_results

                # 将工具结果添加到响应文本中
                if tool_results:
                    tool_results_text = self.tool_executor.format_tool_results(
                        tool_results
                    )
                    generated_text += f"\n\n{tool_results_text}\n\n基于以上工具执行结果，我来回答您的问题："

        # 构建OpenAI格式的响应
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000) % 1000000}",
            "object": "chat.completion",
            "created": int(start_time),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split()),
            },
        }

        # 如果有工具调用，添加到响应中
        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = tool_calls

        # 如果执行了工具，添加执行结果
        if tool_results:
            response["tool_results"] = tool_results

        return response

    def _create_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        model: str,
        tool_names: Optional[List[str]] = None,
        execute_tools: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """流式创建聊天完成"""
        start_time = time.time()
        completion_id = f"chatcmpl-{int(time.time() * 1000) % 1000000}"

        collected_text = ""
        last_tool_calls = []

        # 流式生成
        for chunk in self.llm.generate_stream(prompt, sampling_params, use_tqdm=False):
            # 构建流式响应块
            response_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(start_time),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            }

            if not chunk["finished"]:
                # 增量文本
                if chunk["text"]:
                    response_chunk["choices"][0]["delta"]["content"] = chunk["text"]
                    collected_text += chunk["text"]

                # 检查是否有新的工具调用
                if tool_names and collected_text:
                    current_tool_calls = self._parse_tool_calls_from_text(
                        collected_text, tool_names
                    )
                    if len(current_tool_calls) > len(last_tool_calls):
                        # 有新的工具调用
                        new_calls = current_tool_calls[len(last_tool_calls) :]
                        response_chunk["choices"][0]["delta"]["tool_calls"] = new_calls
                        last_tool_calls = current_tool_calls

            else:
                # 完成
                final_tool_calls = []
                if tool_names and collected_text:
                    final_tool_calls = self._parse_tool_calls_from_text(
                        collected_text, tool_names
                    )

                # 执行工具调用（如果需要）
                if execute_tools and final_tool_calls:
                    tool_results = self.execute_tool_calls(final_tool_calls)
                    if tool_results:
                        # 发送工具执行结果
                        tool_result_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(start_time),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_results": tool_results},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield tool_result_chunk

                if final_tool_calls:
                    response_chunk["choices"][0]["finish_reason"] = "tool_calls"
                else:
                    response_chunk["choices"][0]["finish_reason"] = "stop"

            yield response_chunk


def create_chat_completion_with_tools(llm: LLM) -> ChatCompletionWithTools:
    """
    创建支持工具执行的ChatCompletion实例的便捷函数

    Args:
        llm: LLM实例

    Returns:
        ChatCompletionWithTools实例
    """
    return ChatCompletionWithTools(llm)
