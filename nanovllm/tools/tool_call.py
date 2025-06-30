"""
工具调用相关的数据结构
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import json


class ToolCallStatus(Enum):
    """工具调用状态"""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ToolCall:
    """
    工具调用请求
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    status: ToolCallStatus = ToolCallStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status.value,
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """从字典创建ToolCall对象"""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            status=ToolCallStatus(data.get("status", "pending")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ToolCall":
        """从JSON字符串创建ToolCall对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class ToolCallResult:
    """
    工具调用结果
    """

    tool_call_id: str
    status: ToolCallStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            "tool_call_id": self.tool_call_id,
            "status": self.status.value,
            "execution_time": self.execution_time,
        }

        if self.status == ToolCallStatus.SUCCESS:
            data["result"] = self.result
        elif self.status == ToolCallStatus.ERROR:
            data["error"] = self.error

        return data

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_text(self) -> str:
        """转换为文本格式，用于模型输入"""
        if self.status == ToolCallStatus.SUCCESS:
            return f"工具调用成功: {json.dumps(self.result, ensure_ascii=False)}"
        elif self.status == ToolCallStatus.ERROR:
            return f"工具调用失败: {self.error}"
        else:
            return f"工具调用状态: {self.status.value}"

    @classmethod
    def success(
        cls, tool_call_id: str, result: Any, execution_time: float = 0.0
    ) -> "ToolCallResult":
        """创建成功的工具调用结果"""
        return cls(
            tool_call_id=tool_call_id,
            status=ToolCallStatus.SUCCESS,
            result=result,
            execution_time=execution_time,
        )

    @classmethod
    def error(
        cls, tool_call_id: str, error: str, execution_time: float = 0.0
    ) -> "ToolCallResult":
        """创建失败的工具调用结果"""
        return cls(
            tool_call_id=tool_call_id,
            status=ToolCallStatus.ERROR,
            error=error,
            execution_time=execution_time,
        )
