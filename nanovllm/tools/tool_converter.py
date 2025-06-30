"""
工具转换器

提供将字典格式的工具定义转换为ToolDefinition对象的功能
"""

from typing import Dict, Any, List, Optional, Callable
from .tool_definition import ToolDefinition, Parameter, ParameterType


def dict_to_tool_definition(
    tool_dict: Dict[str, Any], function: Optional[Callable] = None
) -> ToolDefinition:
    """
    将字典类型的工具定义转换成ToolDefinition对象

    Args:
        tool_dict: 字典格式的工具定义，支持OpenAI Function Calling格式
        function: 可选的函数对象

    Returns:
        ToolDefinition对象

    Examples:
        >>> tool_dict = {
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "获取天气信息",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "city": {
        ...                     "type": "string",
        ...                     "description": "城市名称"
        ...                 },
        ...                 "unit": {
        ...                     "type": "string",
        ...                     "description": "温度单位",
        ...                     "enum": ["celsius", "fahrenheit"]
        ...                 }
        ...             },
        ...             "required": ["city"]
        ...         }
        ...     }
        ... }
        >>> tool_def = dict_to_tool_definition(tool_dict)
    """
    # 处理OpenAI Function Calling格式
    if tool_dict.get("type") == "function" and "function" in tool_dict:
        func_def = tool_dict["function"]
        name = func_def.get("name", "")
        description = func_def.get("description", "")
        parameters_schema = func_def.get("parameters", {})
    else:
        # 直接格式
        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        parameters_schema = tool_dict.get("parameters", {})

    # 解析参数
    parameters = []
    if parameters_schema and parameters_schema.get("type") == "object":
        properties = parameters_schema.get("properties", {})
        required_params = set(parameters_schema.get("required", []))

        for param_name, param_schema in properties.items():
            parameter = _parse_parameter(
                param_name, param_schema, param_name in required_params
            )
            parameters.append(parameter)

    return ToolDefinition(
        name=name, description=description, parameters=parameters, function=function
    )


def _parse_parameter(
    name: str, schema: Dict[str, Any], required: bool = True
) -> Parameter:
    """
    解析单个参数的JSON Schema

    Args:
        name: 参数名称
        schema: 参数的JSON Schema
        required: 是否必需

    Returns:
        Parameter对象
    """
    # 解析参数类型
    param_type_str = schema.get("type", "string")
    param_type = _string_to_parameter_type(param_type_str)

    # 基本信息
    description = schema.get("description", f"参数 {name}")
    enum_values = schema.get("enum")
    default_value = schema.get("default")

    # 处理复杂类型
    properties = None
    items = None

    if param_type == ParameterType.OBJECT and "properties" in schema:
        properties = {}
        obj_required = set(schema.get("required", []))
        for prop_name, prop_schema in schema["properties"].items():
            properties[prop_name] = _parse_parameter(
                prop_name, prop_schema, prop_name in obj_required
            )

    if param_type == ParameterType.ARRAY and "items" in schema:
        items = _parse_parameter("item", schema["items"], True)

    return Parameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        default=default_value,
        enum=enum_values,
        properties=properties,
        items=items,
    )


def _string_to_parameter_type(type_str: str) -> ParameterType:
    """
    将字符串类型转换为ParameterType枚举

    Args:
        type_str: 类型字符串

    Returns:
        ParameterType枚举值
    """
    type_mapping = {
        "string": ParameterType.STRING,
        "integer": ParameterType.INTEGER,
        "number": ParameterType.FLOAT,
        "float": ParameterType.FLOAT,
        "boolean": ParameterType.BOOLEAN,
        "array": ParameterType.ARRAY,
        "object": ParameterType.OBJECT,
    }

    return type_mapping.get(type_str.lower(), ParameterType.STRING)


def batch_dict_to_tool_definitions(
    tools_dict_list: List[Dict[str, Any]],
    functions: Optional[Dict[str, Callable]] = None,
) -> List[ToolDefinition]:
    """
    批量转换字典格式的工具定义

    Args:
        tools_dict_list: 字典格式的工具定义列表
        functions: 可选的函数字典，键为函数名，值为函数对象

    Returns:
        ToolDefinition对象列表
    """
    tool_definitions = []
    functions = functions or {}

    for tool_dict in tools_dict_list:
        # 获取工具名称
        if tool_dict.get("type") == "function" and "function" in tool_dict:
            tool_name = tool_dict["function"].get("name", "")
        else:
            tool_name = tool_dict.get("name", "")

        # 获取对应的函数
        function = functions.get(tool_name)

        # 转换为ToolDefinition
        tool_def = dict_to_tool_definition(tool_dict, function)
        tool_definitions.append(tool_def)

    return tool_definitions


def tool_definition_to_dict(
    tool_def: ToolDefinition, format_type: str = "openai"
) -> Dict[str, Any]:
    """
    将ToolDefinition对象转换为字典格式

    Args:
        tool_def: ToolDefinition对象
        format_type: 输出格式，"openai" 或 "simple"

    Returns:
        字典格式的工具定义
    """
    if format_type == "openai":
        return {"type": "function", "function": tool_def.to_function_schema()}
    else:
        # 简单格式
        properties = {}
        required = []

        for param in tool_def.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# 便捷函数
def create_tool_from_openai_format(
    openai_tool: Dict[str, Any], function: Optional[Callable] = None
) -> ToolDefinition:
    """
    从OpenAI格式创建工具定义的便捷函数

    Args:
        openai_tool: OpenAI格式的工具定义
        function: 可选的函数对象

    Returns:
        ToolDefinition对象
    """
    return dict_to_tool_definition(openai_tool, function)


def create_simple_tool(
    name: str,
    description: str,
    parameters: Dict[str, Dict[str, Any]],
    function: Optional[Callable] = None,
) -> ToolDefinition:
    """
    创建简单工具定义的便捷函数

    Args:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义字典
        function: 可选的函数对象

    Returns:
        ToolDefinition对象

    Example:
        >>> tool = create_simple_tool(
        ...     "get_weather",
        ...     "获取天气信息",
        ...     {
        ...         "city": {"type": "string", "description": "城市名称", "required": True},
        ...         "unit": {"type": "string", "description": "温度单位", "required": False}
        ...     }
        ... )
    """
    # 构建OpenAI格式
    properties = {}
    required = []

    for param_name, param_info in parameters.items():
        param_schema = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", f"参数 {param_name}"),
        }

        if "enum" in param_info:
            param_schema["enum"] = param_info["enum"]

        properties[param_name] = param_schema

        if param_info.get("required", True):
            required.append(param_name)

    tool_dict = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return dict_to_tool_definition(tool_dict, function)
