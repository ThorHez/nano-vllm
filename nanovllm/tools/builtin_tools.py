"""
内置工具函数

提供一些常用的工具函数
"""

import os
import json
import requests
import time
import math
import random
from datetime import datetime
from typing import Dict, Any, List
from .tool_definition import ToolDefinition, Parameter, ParameterType, create_function_tool


def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> float:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果
    """
    # 安全的数学表达式计算
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars for c in expression):
        raise ValueError("表达式包含不允许的字符")
    
    # 使用eval计算，但限制可用的内置函数
    allowed_names = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "pow": pow,
        "max": max,
        "min": min,
        "sum": sum,
        "math": math,
    }
    
    try:
        result = eval(expression, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")


def get_weather(city: str) -> Dict[str, Any]:
    """
    获取天气信息（模拟）
    
    Args:
        city: 城市名称
        
    Returns:
        天气信息字典
    """
    # 这是一个模拟的天气API
    # 在实际应用中，应该调用真实的天气API
    
    weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "大雨", "雪"]
    
    return {
        "city": city,
        "temperature": random.randint(-10, 35),
        "condition": random.choice(weather_conditions),
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(0, 20),
        "timestamp": get_current_time()
    }


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    搜索网页（模拟）
    
    Args:
        query: 搜索查询
        num_results: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    # 这是一个模拟的搜索API
    # 在实际应用中，应该调用真实的搜索API
    
    results = []
    for i in range(min(num_results, 10)):
        results.append({
            "title": f"关于'{query}'的搜索结果 {i+1}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"这是关于'{query}'的相关内容摘要..."
        })
    
    return results


def read_file(file_path: str) -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"读取文件失败: {str(e)}")


def write_file(file_path: str, content: str) -> str:
    """
    写入文件
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
        
    Returns:
        操作结果
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件: {file_path}"
    except Exception as e:
        raise ValueError(f"写入文件失败: {str(e)}")


def generate_random_number(min_val: int = 1, max_val: int = 100) -> int:
    """
    生成随机数
    
    Args:
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        随机数
    """
    return random.randint(min_val, max_val)


def format_json(data: str) -> str:
    """
    格式化JSON字符串
    
    Args:
        data: JSON字符串
        
    Returns:
        格式化后的JSON字符串
    """
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {str(e)}")


def get_builtin_tools() -> Dict[str, ToolDefinition]:
    """
    获取所有内置工具
    
    Returns:
        内置工具字典
    """
    tools = {}
    
    # 时间工具
    tools["get_current_time"] = ToolDefinition(
        name="get_current_time",
        description="获取当前时间",
        parameters=[],
        function=get_current_time
    )
    
    # 计算器工具
    tools["calculate"] = ToolDefinition(
        name="calculate",
        description="计算数学表达式",
        parameters=[
            Parameter(
                name="expression",
                type=ParameterType.STRING,
                description="要计算的数学表达式",
                required=True
            )
        ],
        function=calculate
    )
    
    # 天气工具
    tools["get_weather"] = ToolDefinition(
        name="get_weather",
        description="获取指定城市的天气信息",
        parameters=[
            Parameter(
                name="city",
                type=ParameterType.STRING,
                description="城市名称",
                required=True
            )
        ],
        function=get_weather
    )
    
    # 搜索工具
    tools["search_web"] = ToolDefinition(
        name="search_web",
        description="搜索网页内容",
        parameters=[
            Parameter(
                name="query",
                type=ParameterType.STRING,
                description="搜索查询",
                required=True
            ),
            Parameter(
                name="num_results",
                type=ParameterType.INTEGER,
                description="返回结果数量",
                required=False,
                default=5
            )
        ],
        function=search_web
    )
    
    # 文件读取工具
    tools["read_file"] = ToolDefinition(
        name="read_file",
        description="读取文件内容",
        parameters=[
            Parameter(
                name="file_path",
                type=ParameterType.STRING,
                description="文件路径",
                required=True
            )
        ],
        function=read_file
    )
    
    # 文件写入工具
    tools["write_file"] = ToolDefinition(
        name="write_file",
        description="写入文件内容",
        parameters=[
            Parameter(
                name="file_path",
                type=ParameterType.STRING,
                description="文件路径",
                required=True
            ),
            Parameter(
                name="content",
                type=ParameterType.STRING,
                description="要写入的内容",
                required=True
            )
        ],
        function=write_file
    )
    
    # 随机数生成工具
    tools["generate_random_number"] = ToolDefinition(
        name="generate_random_number",
        description="生成指定范围内的随机数",
        parameters=[
            Parameter(
                name="min_val",
                type=ParameterType.INTEGER,
                description="最小值",
                required=False,
                default=1
            ),
            Parameter(
                name="max_val",
                type=ParameterType.INTEGER,
                description="最大值",
                required=False,
                default=100
            )
        ],
        function=generate_random_number
    )
    
    # JSON格式化工具
    tools["format_json"] = ToolDefinition(
        name="format_json",
        description="格式化JSON字符串",
        parameters=[
            Parameter(
                name="data",
                type=ParameterType.STRING,
                description="要格式化的JSON字符串",
                required=True
            )
        ],
        function=format_json
    )
    
    return tools


def register_custom_tool(func, name: str = None, description: str = None) -> ToolDefinition:
    """
    注册自定义工具函数
    
    Args:
        func: Python函数
        name: 工具名称（可选，默认使用函数名）
        description: 工具描述（可选，默认使用函数文档字符串）
        
    Returns:
        工具定义
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or f"自定义工具: {tool_name}"
    
    return create_function_tool(tool_name, tool_description, func) 