#!/usr/bin/env python3
"""
API客户端示例

展示如何使用HTTP API调用nanovllm服务
"""

import requests
import json
import time


def test_basic_chat():
    """测试基础聊天"""
    print("🔵 测试基础聊天")
    
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions', 
        json={
            "messages": [
                {"role": "user", "content": "你好，请介绍一下你自己"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 响应: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ 错误: {response.text}")


def test_tool_calling():
    """测试工具调用"""
    print("\n🔧 测试工具调用")
    
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
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions',
        json={
            "messages": [
                {"role": "user", "content": "请计算 25 * 8 + 15 的结果"}
            ],
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": 150
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        message = result['choices'][0]['message']
        print(f"✅ AI回复: {message['content']}")
        
        if message.get('tool_calls'):
            print("🔧 检测到工具调用:")
            for call in message['tool_calls']:
                func = call['function']
                print(f"  • {func['name']}({func['arguments']})")
    else:
        print(f"❌ 错误: {response.text}")


def test_streaming():
    """测试流式响应"""
    print("\n🌊 测试流式响应")
    
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions',
        json={
            "messages": [
                {"role": "user", "content": "请写一首关于春天的短诗"}
            ],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": True
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("✅ 流式响应: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 移除 "data: " 前缀
                    if data == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data)
                        choice = chunk['choices'][0]
                        if 'content' in choice['delta']:
                            print(choice['delta']['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        
        print("\n")
    else:
        print(f"❌ 错误: {response.text}")


def test_tool_choice_strategies():
    """测试工具选择策略"""
    print("\n🎛️  测试工具选择策略")
    
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
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    strategies = [
        {"tool_choice": "auto", "desc": "自动选择"},
        {"tool_choice": "none", "desc": "不使用工具"},
        {"tool_choice": {"type": "function", "function": {"name": "calculate"}}, "desc": "强制使用计算器"}
    ]
    
    for strategy in strategies:
        print(f"\n📋 测试策略: {strategy['desc']}")
        
        response = requests.post('http://127.0.0.1:8000/v1/chat/completions',
            json={
                "messages": [
                    {"role": "user", "content": "10 加 20 等于多少？"}
                ],
                "tools": tools,
                "tool_choice": strategy["tool_choice"],
                "temperature": 0.3,
                "max_tokens": 100
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']
            print(f"  回复: {message['content']}")
            print(f"  完成原因: {result['choices'][0]['finish_reason']}")
            
            if message.get('tool_calls'):
                print(f"  工具调用: {len(message['tool_calls'])}个")
        else:
            print(f"  ❌ 错误: {response.text}")


def test_multi_turn_conversation():
    """测试多轮对话"""
    print("\n💬 测试多轮对话")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # 第一轮：用户询问
    messages = [
        {"role": "user", "content": "请查看北京的天气"}
    ]
    
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions',
        json={
            "messages": messages,
            "tools": tools,
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        assistant_msg = result['choices'][0]['message']
        messages.append(assistant_msg)
        
        print(f"🤖 AI: {assistant_msg['content']}")
        
        if assistant_msg.get('tool_calls'):
            print("🔧 工具调用检测:")
            for call in assistant_msg['tool_calls']:
                func = call['function']
                print(f"  • {func['name']}({func['arguments']})")
            
            # 模拟工具执行结果
            tool_result = {
                "role": "tool",
                "tool_call_id": assistant_msg['tool_calls'][0]['id'],
                "content": "北京今天天气：晴天，温度22°C，空气质量良好"
            }
            messages.append(tool_result)
            
            # 第二轮：基于工具结果继续对话
            response2 = requests.post('http://127.0.0.1:8000/v1/chat/completions',
                json={
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            
            if response2.status_code == 200:
                result2 = response2.json()
                final_msg = result2['choices'][0]['message']
                print(f"🤖 AI (基于工具结果): {final_msg['content']}")
    else:
        print(f"❌ 错误: {response.text}")


def main():
    print("🚀 nanovllm API客户端测试")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        response = requests.get('http://127.0.0.1:8000/health')
        if response.status_code == 200:
            print("✅ 服务器运行正常")
        else:
            print("❌ 服务器响应异常")
            return
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保API服务器正在运行")
        print("💡 运行命令: python api_server.py")
        return
    
    # 运行测试
    test_basic_chat()
    test_tool_calling()
    test_streaming()
    test_tool_choice_strategies()
    test_multi_turn_conversation()
    
    print("\n✨ 所有测试完成！")


if __name__ == "__main__":
    main() 