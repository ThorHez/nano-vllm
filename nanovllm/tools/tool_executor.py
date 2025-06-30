"""
工具执行器

负责执行工具调用并管理工具状态
"""

import time
import json
import uuid
import traceback
from typing import Dict, Any, List, Optional, Callable
from .tool_definition import ToolDefinition
from .tool_call import ToolCall
from .tool_parser import ToolCallParser
from .builtin_tools import get_builtin_tools
import hashlib


class ToolExecutor:
    """工具执行器"""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.parser = ToolCallParser()
        self.execution_history: List[Dict[str, Any]] = []

        # 加载内置工具
        self.load_builtin_tools()

    def load_builtin_tools(self):
        """加载内置工具"""
        builtin_tools = get_builtin_tools()
        self.tools.update(builtin_tools)

    def register_tool(self, tool: ToolDefinition):
        """注册工具"""
        self.tools[tool.name] = tool

    def register_function(
        self, func: Callable, name: str = None, description: str = None
    ):
        """注册函数作为工具"""
        from .builtin_tools import register_custom_tool

        tool = register_custom_tool(func, name, description)
        self.register_tool(tool)

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())

    def contains_tool_call(self, text: str) -> bool:
        """检查文本是否包含工具调用"""
        return self.parser.contains_tool_call(text)

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """解析文本中的工具调用"""
        available_tools = self.get_available_tools()
        return self.parser.parse(text, available_tools)

    def execute_tool_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        """
        执行单个工具调用

        Args:
            tool_call: 工具调用对象

        Returns:
            执行结果字典，包含：
            - success: 是否成功
            - result: 执行结果
            - error: 错误信息（如果有）
            - execution_time: 执行时间
        """

        start_time = time.time()
        execution_id = str(uuid.uuid4())

        result = {
            "id": execution_id,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "success": False,
            "result": None,
            "error": None,
            "execution_time": 0.0,
            "timestamp": time.time(),
        }

        try:
            # 检查工具是否存在
            if tool_call.name not in self.tools:
                raise ValueError(f"工具 '{tool_call.name}' 不存在")

            tool = self.tools[tool_call.name]

            # 验证参数
            self._validate_arguments(tool, tool_call.arguments)

            # 执行工具函数
            if tool.function:
                if tool_call.arguments:
                    tool_result = tool.function(**tool_call.arguments)
                else:
                    tool_result = tool.function()
            else:
                raise ValueError(f"工具 '{tool_call.name}' 没有可执行的函数")

            result["success"] = True
            result["result"] = tool_result

        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        finally:
            result["execution_time"] = time.time() - start_time
            self.execution_history.append(result.copy())

        return result

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """
        执行多个工具调用

        Args:
            tool_calls: 工具调用列表

        Returns:
            执行结果列表
        """
        results = []
        for tool_call in tool_calls:
            result = self.execute_tool_call(tool_call)
            results.append(result)

        return results

    def parse_and_execute(self, text: str, tool_call_hash_set: set):
        """
        解析并执行文本中的工具调用

        Args:
            text: 包含工具调用的文本，格式如：
                 '<think>...</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'

        Returns:
            被<tool_call></tool_call>包围的工具执行结果字符串
        """
        tool_calls = self.parse_tool_calls(text)

        tool_call_without_id = tool_calls[-1].to_dict()
        tool_call_without_id.pop("id")

        tool_calls_hash = hashlib.md5(
            json.dumps(
                tool_call_without_id, ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest()
        if tool_calls_hash in tool_call_hash_set:
            return None, None
        else:
            tool_call_hash_set.add(tool_calls_hash)

        if tool_calls:
            # 只取最后一个工具，因为这种场景下只可能调用最后一个工具，因为前边的工具已调用完毕
            tool_calls = [tool_calls[-1]]
            print(f"tool_calls: {tool_calls}")
        results = self.execute_tool_calls(tool_calls)
        return tool_calls, results

    def format_tool_result(self, result: Dict[str, Any]) -> str:
        """
        格式化工具执行结果为文本

        Args:
            result: 工具执行结果

        Returns:
            格式化的结果文本
        """
        if result["success"]:
            tool_result = result["result"]
            if isinstance(tool_result, (dict, list)):
                tool_result = json.dumps(tool_result, ensure_ascii=False, indent=2)
            elif not isinstance(tool_result, str):
                tool_result = str(tool_result)

            return tool_result
        else:
            return f"工具调用失败: {result['tool_name']}\n错误: {result['error']}"

    def format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化多个工具执行结果

        Args:
            results: 工具执行结果列表

        Returns:
            格式化的结果文本
        """
        formatted_results = []
        for result in results:
            formatted_results.append(self.format_tool_result(result))

        return "\n\n".join(formatted_results)

    def _validate_arguments(self, tool: ToolDefinition, arguments: Dict[str, Any]):
        """验证工具参数"""
        if not tool.parameters:
            return

        # 检查必需参数
        required_params = [p.name for p in tool.parameters if p.required]
        for param_name in required_params:
            if param_name not in arguments:
                raise ValueError(f"缺少必需参数: {param_name}")

        # 检查参数类型（简单验证）
        for param in tool.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                # 这里可以添加更详细的类型验证
                if param.type.value == "string" and not isinstance(value, str):
                    try:
                        arguments[param.name] = str(value)
                    except:
                        raise ValueError(f"参数 {param.name} 应为字符串类型")
                elif param.type.value == "integer" and not isinstance(value, int):
                    try:
                        arguments[param.name] = int(value)
                    except:
                        raise ValueError(f"参数 {param.name} 应为整数类型")
                elif param.type.value == "number" and not isinstance(
                    value, (int, float)
                ):
                    try:
                        arguments[param.name] = float(value)
                    except:
                        raise ValueError(f"参数 {param.name} 应为数字类型")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history.copy()

    def clear_execution_history(self):
        """清空执行历史"""
        self.execution_history.clear()

    def get_tool_definitions_for_prompt(self) -> str:
        """
        获取工具定义的文本描述，用于添加到prompt中

        Returns:
            工具定义的文本描述
        """
        if not self.tools:
            return ""

        tool_descriptions = []
        tool_descriptions.append("可用工具:")

        for tool_name, tool in self.tools.items():
            desc = f"- {tool_name}: {tool.description}"
            if tool.parameters:
                params = []
                for param in tool.parameters:
                    param_desc = f"{param.name}({param.type.value})"
                    if param.required:
                        param_desc += "*"
                    if param.description:
                        param_desc += f" - {param.description}"
                    params.append(param_desc)
                desc += f"\n  参数: {', '.join(params)}"
            tool_descriptions.append(desc)

        tool_descriptions.append(
            '\n工具调用格式: {"tool_call": {"name": "工具名", "arguments": {参数字典}}}'
        )

        return "\n".join(tool_descriptions)
