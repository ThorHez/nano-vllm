#!/usr/bin/env python3
"""
测试ChatCompletion功能
"""

import json
from nanovllm import LLM
from nanovllm.chat_completion import ChatCompletion
from nanovllm.tools import create_function_tool


def test_basic_chat():
    """测试基本聊天功能"""
    print("=" * 50)
    print("测试基本聊天功能")
    print("=" * 50)
    
    # 初始化模型
    llm = LLM("models/Qwen3-0.6B")
    chat = ChatCompletion(llm)
    
    messages = [
        {"role": "user", "content": "你好，请简单介绍一下你自己"}
    ]
    
    response = chat.create(
        messages=messages,
        max_tokens=100,
        temperature=0.8
    )
    
    print("响应:")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def test_tool_calling():
    """测试工具调用功能"""
    print("=" * 50)
    print("测试工具调用功能")
    print("=" * 50)
    
    # 初始化模型
    llm = LLM("models/Qwen3-0.6B")
    chat = ChatCompletion(llm)
    
    # 定义计算器工具
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    messages = [
        {"role": "user", "content": "请计算 15 + 27 * 3 的结果"}
    ]
    
    response = chat.create(
        messages=messages,
        tools=[calculator_tool],
        max_tokens=150,
        temperature=0.3
    )
    
    print("响应:")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def test_streaming_chat():
    """测试流式聊天"""
    print("=" * 50)
    print("测试流式聊天")
    print("=" * 50)
    
    # 初始化模型
    llm = LLM("models/Qwen3-0.6B")
    chat = ChatCompletion(llm)
    
    messages = [
        {"role": "user", "content": "请写一首关于编程的短诗"}
    ]
    
    print("流式响应:")
    for chunk in chat.create(
        messages=messages,
        max_tokens=200,
        temperature=0.8,
        stream=True
    ):
        print(json.dumps(chunk, indent=2, ensure_ascii=False))
        print("-" * 30)


if __name__ == "__main__":
    try:
        # 运行测试
        test_basic_chat()
        print("\n" + "=" * 80 + "\n")
        
        test_tool_calling()
        print("\n" + "=" * 80 + "\n")
        
        test_streaming_chat()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 