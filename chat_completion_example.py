#!/usr/bin/env python3
"""
OpenAI风格ChatCompletion接口示例

展示如何使用ChatCompletion接口进行工具调用
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from nanovllm import LLM
from nanovllm.chat_completion import create_chat_completion
import json


def main():
    print("🚀 OpenAI风格ChatCompletion接口示例")
    print("=" * 50)
    
    # 初始化LLM
    model_path = "/root/models/Qwen3-0.6B"
    print(f"📚 加载模型: {model_path}")
    
    llm = LLM(model_path, device="cpu")
    
    # 创建ChatCompletion实例
    chat_completion = create_chat_completion(llm)
    
    # 定义工具（OpenAI格式）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式，如 '2+3*4'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "unit": {
                            "type": "string",
                            "description": "温度单位",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    print("🛠️  定义的工具:")
    for tool in tools:
        func = tool["function"]
        print(f"  • {func['name']}: {func['description']}")
    
    print("\n" + "="*50)
    print("示例1: 基础工具调用")
    print("="*50)
    
    # 示例1: 数学计算
    messages = [
        {"role": "user", "content": "请帮我计算 15 * 8 + 32 的结果"}
    ]
    
    print("👤 用户:", messages[0]["content"])
    
    response = chat_completion.create(
        messages=messages,
        tools=tools,
        temperature=0.3,
        max_tokens=150
    )
    
    print("🤖 AI回复:")
    print(f"  内容: {response['choices'][0]['message']['content']}")
    print(f"  完成原因: {response['choices'][0]['finish_reason']}")
    
    # 检查工具调用
    if response['choices'][0]['message'].get('tool_calls'):
        print("🔧 检测到工具调用:")
        for call in response['choices'][0]['message']['tool_calls']:
            func = call['function']
            print(f"  • ID: {call['id']}")
            print(f"  • 工具: {func['name']}")
            print(f"  • 参数: {func['arguments']}")
    
    print("\n" + "="*50)
    print("示例2: 多轮对话（包含工具调用）")
    print("="*50)
    
    # 示例2: 多轮对话
    conversation = [
        {"role": "user", "content": "请查看北京的天气，然后帮我计算一下如果温度是25度，转换为华氏度是多少"}
    ]
    
    print("👤 用户:", conversation[0]["content"])
    
    response = chat_completion.create(
        messages=conversation,
        tools=tools,
        temperature=0.7,
        max_tokens=200
    )
    
    assistant_msg = response['choices'][0]['message']
    conversation.append(assistant_msg)
    
    print("🤖 AI回复:")
    print(f"  内容: {assistant_msg['content']}")
    
    if assistant_msg.get('tool_calls'):
        print("🔧 工具调用:")
        for call in assistant_msg['tool_calls']:
            func = call['function']
            print(f"  • {func['name']}({func['arguments']})")
        
        # 模拟工具执行结果
        tool_results = []
        for call in assistant_msg['tool_calls']:
            func = call['function']
            if func['name'] == 'get_weather':
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call['id'],
                    "content": "北京今天天气：晴天，气温25°C，湿度60%"
                })
            elif func['name'] == 'calculate':
                # 解析参数
                args = json.loads(func['arguments'])
                if '25*9/5+32' in args.get('expression', ''):
                    tool_results.append({
                        "role": "tool", 
                        "tool_call_id": call['id'],
                        "content": "计算结果: 77.0"
                    })
        
        # 添加工具结果到对话
        conversation.extend(tool_results)
        
        # 继续对话，让AI基于工具结果回答
        final_response = chat_completion.create(
            messages=conversation,
            temperature=0.7,
            max_tokens=100
        )
        
        print("🛠️  工具执行后的AI回复:")
        print(f"  {final_response['choices'][0]['message']['content']}")
    
    print("\n" + "="*50)
    print("示例3: 流式工具调用")
    print("="*50)
    
    messages = [
        {"role": "user", "content": "请计算 100 / 4，然后告诉我结果"}
    ]
    
    print("👤 用户:", messages[0]["content"])
    print("🤖 AI (流式): ", end="", flush=True)
    
    # 流式调用
    stream = chat_completion.create(
        messages=messages,
        tools=tools,
        temperature=0.3,
        stream=True
    )
    
    collected_content = ""
    tool_calls_detected = []
    
    for chunk in stream:
        choice = chunk['choices'][0]
        delta = choice['delta']
        
        # 输出文本内容
        if 'content' in delta and delta['content']:
            print(delta['content'], end='', flush=True)
            collected_content += delta['content']
        
        # 检测工具调用
        if 'tool_calls' in delta:
            tool_calls_detected.extend(delta['tool_calls'])
        
        # 检查是否完成
        if choice['finish_reason']:
            print(f"\n⭐ 完成原因: {choice['finish_reason']}")
            break
    
    if tool_calls_detected:
        print("🔧 流式检测到的工具调用:")
        for call in tool_calls_detected:
            func = call['function']
            print(f"  • {func['name']}({func['arguments']})")
    
    print("\n" + "="*50)
    print("示例4: 工具选择策略")
    print("="*50)
    
    test_cases = [
        {"tool_choice": "auto", "description": "自动选择"},
        {"tool_choice": "none", "description": "不使用工具"},
        {"tool_choice": {"type": "function", "function": {"name": "calculate"}}, "description": "强制使用计算器"}
    ]
    
    for case in test_cases:
        print(f"\n🎛️  测试 {case['description']}:")
        
        response = chat_completion.create(
            messages=[{"role": "user", "content": "10 + 20 等于多少？"}],
            tools=tools,
            tool_choice=case["tool_choice"],
            temperature=0.3,
            max_tokens=100
        )
        
        msg = response['choices'][0]['message']
        print(f"  AI回复: {msg['content']}")
        print(f"  完成原因: {response['choices'][0]['finish_reason']}")
        
        if msg.get('tool_calls'):
            print(f"  工具调用: {len(msg['tool_calls'])}个")
    
    print("\n✨ 所有示例完成！")


if __name__ == "__main__":
    main() 