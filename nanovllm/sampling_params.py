"""
采样参数配置

定义生成过程中的各种参数，包括温度、最大token数、工具调用设置等
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    采样参数类，用于控制文本生成过程
    
    Attributes:
        temperature: 采样温度，控制随机性 (0.0-2.0)
        max_tokens: 最大生成token数
        ignore_eos: 是否忽略结束符
        tools: 可用工具列表
        tool_choice: 工具选择策略
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    
    def __post_init__(self):
        """参数验证"""
        if self.temperature < 0.0:
            raise ValueError("temperature必须大于等于0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens必须大于0")