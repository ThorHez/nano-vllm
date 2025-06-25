"""
工具调用解析器

从模型输出中解析工具调用
"""

import re
import json
import uuid
from typing import List, Optional, Dict, Any
from .tool_call import ToolCall


class ToolCallParser:
    """
    工具调用解析器
    
    支持多种格式的工具调用解析：
    1. JSON格式：{"tool_call": {"name": "函数名", "arguments": {...}}}
    2. 函数调用格式：function_name(arg1="value1", arg2="value2")
    3. XML格式：<tool_call name="函数名"><arguments>...</arguments></tool_call>
    """
    
    def __init__(self):
        # JSON格式的正则表达式
        self.json_pattern = re.compile(
            r'\{[^{}]*"tool_call"[^{}]*\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}[^{}]*\}',
            re.DOTALL | re.IGNORECASE
        )
        
        # 函数调用格式的正则表达式
        self.function_pattern = re.compile(
            r'(\w+)\s*\(([^)]*)\)',
            re.DOTALL
        )
        
        # XML格式的正则表达式
        self.xml_pattern = re.compile(
            r'<tool_call\s+name="([^"]+)"\s*>(.*?)</tool_call>',
            re.DOTALL | re.IGNORECASE
        )
        
        # 工具调用标记
        self.tool_call_markers = [
            "tool_call",
            "function_call", 
            "调用工具",
            "使用工具",
            "执行函数"
        ]
    
    def parse(self, text: str, available_tools: List[str] = None) -> List[ToolCall]:
        """
        解析文本中的工具调用
        
        Args:
            text: 要解析的文本
            available_tools: 可用的工具名称列表
            
        Returns:
            解析出的工具调用列表
        """
        tool_calls = []
        
        # 尝试JSON格式解析
        json_calls = self._parse_json_format(text, available_tools)
        tool_calls.extend(json_calls)
        
        # 尝试函数调用格式解析
        function_calls = self._parse_function_format(text, available_tools)
        tool_calls.extend(function_calls)
        
        # 尝试XML格式解析
        xml_calls = self._parse_xml_format(text, available_tools)
        tool_calls.extend(xml_calls)
        
        return tool_calls
    
    def _parse_json_format(self, text: str, available_tools: List[str] = None) -> List[ToolCall]:
        """解析JSON格式的工具调用"""
        tool_calls = []
        
        # 查找所有可能的JSON工具调用
        matches = self.json_pattern.findall(text)
        
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_call" in data:
                    tool_data = data["tool_call"]
                    name = tool_data.get("name", "")
                    arguments = tool_data.get("arguments", {})
                    
                    if available_tools and name not in available_tools:
                        continue
                    
                    tool_call = ToolCall(
                        id=str(uuid.uuid4()),
                        name=name,
                        arguments=arguments
                    )
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError):
                continue
        
        return tool_calls
    
    def _parse_function_format(self, text: str, available_tools: List[str] = None) -> List[ToolCall]:
        """解析函数调用格式的工具调用"""
        tool_calls = []
        
        # 查找所有可能的函数调用
        matches = self.function_pattern.findall(text)
        
        for func_name, args_str in matches:
            if available_tools and func_name not in available_tools:
                continue
            
            # 解析参数
            arguments = self._parse_function_arguments(args_str)
            
            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                name=func_name,
                arguments=arguments
            )
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def _parse_xml_format(self, text: str, available_tools: List[str] = None) -> List[ToolCall]:
        """解析XML格式的工具调用"""
        tool_calls = []
        
        matches = self.xml_pattern.findall(text)
        
        for name, args_content in matches:
            if available_tools and name not in available_tools:
                continue
            
            # 尝试解析参数内容
            arguments = {}
            try:
                # 如果是JSON格式
                if args_content.strip().startswith('{'):
                    arguments = json.loads(args_content.strip())
                else:
                    # 尝试解析键值对格式
                    arguments = self._parse_key_value_arguments(args_content)
            except:
                arguments = {"content": args_content.strip()}
            
            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                name=name,
                arguments=arguments
            )
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def _parse_function_arguments(self, args_str: str) -> Dict[str, Any]:
        """解析函数参数字符串"""
        arguments = {}
        
        if not args_str.strip():
            return arguments
        
        # 简单的参数解析（支持key=value格式）
        # 这里可以根据需要扩展支持更复杂的格式
        try:
            # 尝试作为JSON解析
            if args_str.strip().startswith('{'):
                arguments = json.loads(args_str)
            else:
                # 解析key=value格式
                arguments = self._parse_key_value_arguments(args_str)
        except:
            # 如果解析失败，将整个字符串作为单个参数
            arguments = {"args": args_str.strip()}
        
        return arguments
    
    def _parse_key_value_arguments(self, args_str: str) -> Dict[str, Any]:
        """解析键值对参数"""
        arguments = {}
        
        # 简单的键值对解析
        # 支持 key="value", key='value', key=value 格式
        pattern = re.compile(r'(\w+)\s*=\s*(["\']?)([^,"\']*)(["\']?)')
        matches = pattern.findall(args_str)
        
        for key, quote1, value, quote2 in matches:
            # 尝试转换数据类型
            try:
                if value.lower() == 'true':
                    arguments[key] = True
                elif value.lower() == 'false':
                    arguments[key] = False
                elif value.isdigit():
                    arguments[key] = int(value)
                elif value.replace('.', '').isdigit():
                    arguments[key] = float(value)
                else:
                    arguments[key] = value
            except:
                arguments[key] = value
        
        return arguments
    
    def contains_tool_call(self, text: str) -> bool:
        """检查文本是否包含工具调用"""
        # 检查是否包含工具调用标记
        for marker in self.tool_call_markers:
            if marker.lower() in text.lower():
                return True
        
        # 检查是否匹配任何格式的模式
        if (self.json_pattern.search(text) or 
            self.xml_pattern.search(text) or
            self.function_pattern.search(text)):
            return True
        
        return False
    
    def extract_tool_call_text(self, text: str) -> Optional[str]:
        """提取包含工具调用的文本部分"""
        # 查找第一个工具调用的位置
        earliest_pos = len(text)
        
        for match in self.json_pattern.finditer(text):
            earliest_pos = min(earliest_pos, match.start())
        
        for match in self.xml_pattern.finditer(text):
            earliest_pos = min(earliest_pos, match.start())
        
        for match in self.function_pattern.finditer(text):
            earliest_pos = min(earliest_pos, match.start())
        
        if earliest_pos < len(text):
            return text[earliest_pos:]
        
        return None 