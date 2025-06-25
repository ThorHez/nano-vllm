"""
工具定义相关的数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import inspect


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class Parameter:
    """
    工具参数定义
    """
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    properties: Optional[Dict[str, 'Parameter']] = None  # 用于object类型
    items: Optional['Parameter'] = None  # 用于array类型
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema格式"""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
            
        if self.type == ParameterType.OBJECT and self.properties:
            schema["properties"] = {
                name: param.to_json_schema() 
                for name, param in self.properties.items()
            }
            required_props = [
                name for name, param in self.properties.items() 
                if param.required
            ]
            if required_props:
                schema["required"] = required_props
                
        if self.type == ParameterType.ARRAY and self.items:
            schema["items"] = self.items.to_json_schema()
            
        return schema


@dataclass
class ToolDefinition:
    """
    工具定义
    """
    name: str
    description: str
    parameters: List[Parameter] = field(default_factory=list)
    function: Optional[Callable] = None
    examples: Optional[List[str]] = None
    
    def to_function_schema(self) -> Dict[str, Any]:
        """转换为OpenAI Function Calling格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_prompt_format(self) -> str:
        """转换为提示词格式"""
        param_descriptions = []
        for param in self.parameters:
            req_str = "必需" if param.required else "可选"
            param_desc = f"- {param.name} ({param.type.value}, {req_str}): {param.description}"
            if param.default is not None:
                param_desc += f" (默认: {param.default})"
            param_descriptions.append(param_desc)
        
        return f"""工具名称: {self.name}
                    描述: {self.description}
                    参数:
                    {chr(10).join(param_descriptions) if param_descriptions else "无参数"}"""

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """验证参数"""
        for param in self.parameters:
            if param.required and param.name not in params:
                return False
            
            if param.name in params:
                value = params[param.name]
                # 这里可以添加更详细的类型验证
                if param.type == ParameterType.STRING and not isinstance(value, str):
                    return False
                elif param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return False
                elif param.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
                    return False
                elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return False
                    
        return True


def create_function_tool(name: str, description: str, function: Callable) -> ToolDefinition:
    """
    从Python函数自动创建工具定义
    
    Args:
        name: 工具名称
        description: 工具描述
        function: Python函数
        
    Returns:
        工具定义对象
    """
    # 获取函数签名
    sig = inspect.signature(function)
    parameters = []
    
    # 解析函数参数
    for param_name, param in sig.parameters.items():
        # 确定参数类型
        param_type = ParameterType.STRING  # 默认为字符串
        param_desc = f"参数 {param_name}"
        
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == str:
                param_type = ParameterType.STRING
            elif param.annotation == int:
                param_type = ParameterType.INTEGER
            elif param.annotation == float:
                param_type = ParameterType.FLOAT
            elif param.annotation == bool:
                param_type = ParameterType.BOOLEAN
            elif param.annotation == list:
                param_type = ParameterType.ARRAY
            elif param.annotation == dict:
                param_type = ParameterType.OBJECT
        
        # 确定是否必需
        required = param.default == inspect.Parameter.empty
        default_value = None if required else param.default
        
        # 从文档字符串中提取参数描述
        if function.__doc__:
            doc_lines = function.__doc__.split('\n')
            for line in doc_lines:
                line = line.strip()
                if line.startswith(f"{param_name}:") or line.startswith(f"{param_name} "):
                    param_desc = line.split(':', 1)[-1].strip()
                    break
        
        parameters.append(
            Parameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=required,
                default=default_value
            )
        )
    
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        function=function
    ) 