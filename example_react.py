from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
import random
import time
import statistics
from nanovllm.tools.builtin_tools import get_current_time


@tool
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


def create_custom_llm():
    """创建自定义LLM实例"""
    # 使用字符串并忽略类型检查
    return ChatOpenAI(
        base_url="http://10.239.121.23:8002/v1",
        api_key="EMPTY",  # type: ignore
        model="qwen3_32b",
        temperature=0.0
    )


def example_react_single():
    """单次执行React Agent"""
    print("🤖 初始化React Agent...")
    
    # 创建LLM实例
    llm = create_custom_llm()
    
    # 定义工具列表
    tools = [
        get_weather,
    ]
    
    # 创建React Agent
    agent = create_react_agent(llm, tools)
    
    print("✅ React Agent创建成功！")
    print("\n可用工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # 执行单次查询
    response = agent.invoke({"messages": [("user", "What is the weather in Beijing and Tokyo?")]})
    print(f"\n🤖 响应: {response}")
    
    return response


def example_react():
    """统计example_react调用平均时长和AIMessage响应长度"""
    print("📊 开始统计React Agent调用时长和AIMessage响应长度...")
    print("=" * 60)
    
    # 测试参数
    test_runs = 10  # 测试次数
    queries = [
        "What is the weather in Beijing and Tokyo?",
    ]
    
    execution_times = []
    response_lengths = []  # 记录AIMessage响应长度
    ai_message_counts = []  # 记录AIMessage数量
    
    # 预热：先创建一次Agent避免初始化时间影响测试
    print("🔥 预热Agent...")
    llm = create_custom_llm()
    tools = [get_weather]
    agent = create_react_agent(llm, tools)
    
    print(f"🚀 开始执行 {test_runs} 次测试...")
    
    for i in range(test_runs):
        query = queries[i % len(queries)]
        print(f"\n📝 测试 {i+1}/{test_runs}: {query}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 执行Agent调用
            response = agent.invoke({"messages": [("user", query)]})
            
            # 记录结束时间
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # 记录响应长度 - 统计AIMessage的总长度
            ai_message_length = 0
            ai_message_count = 0
            
            if 'messages' in response:
                for message in response['messages']:
                    # 检查是否是AIMessage
                    is_ai_message = False
                    if hasattr(message, 'type') and message.type == 'ai':
                        is_ai_message = True
                    elif hasattr(message, '__class__') and 'AIMessage' in str(message.__class__):
                        is_ai_message = True
                    
                    if is_ai_message:
                        ai_message_count += 1
                        content = getattr(message, 'content', '')
                        ai_message_length += len(content)
            
            response_lengths.append(ai_message_length)
            ai_message_counts.append(ai_message_count)
            
            print(f"⏱️  执行时间: {execution_time:.3f}秒")
            print(f"📄 AIMessage长度: {ai_message_length} 字符 ({ai_message_count} 条AIMessage)")
            
            # 调试信息（可选）
            if i == 0:  # 只在第一次执行时显示调试信息
                print(f"🔍 调试信息: 响应包含 {len(response.get('messages', []))} 条消息")
                for j, msg in enumerate(response.get('messages', [])):
                    msg_type = getattr(msg, 'type', 'unknown')
                    msg_class = str(msg.__class__.__name__)
                    msg_length = len(getattr(msg, 'content', ''))
                    print(f"  消息{j+1}: {msg_class} (type: {msg_type}, 长度: {msg_length})")
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            continue
    
    # 统计分析
    if execution_times and response_lengths and ai_message_counts:
        print("\n" + "=" * 60)
        print("📊 统计结果:")
        print(f"✅ 成功执行次数: {len(execution_times)}")
        
        # 执行时间统计
        print(f"\n⏱️  时间统计:")
        print(f"  平均执行时间: {statistics.mean(execution_times):.3f}秒")
        print(f"  最快执行时间: {min(execution_times):.3f}秒")
        print(f"  最慢执行时间: {max(execution_times):.3f}秒")
        
        # 响应长度统计
        print(f"\n📄 AIMessage长度统计:")
        print(f"  平均AIMessage长度: {statistics.mean(response_lengths):.1f} 字符")
        print(f"  最短AIMessage长度: {min(response_lengths)} 字符")
        print(f"  最长AIMessage长度: {max(response_lengths)} 字符")
        
        # AIMessage数量统计
        print(f"\n📊 AIMessage数量统计:")
        print(f"  平均AIMessage数量: {statistics.mean(ai_message_counts):.1f} 条")
        print(f"  最少AIMessage数量: {min(ai_message_counts)} 条")
        print(f"  最多AIMessage数量: {max(ai_message_counts)} 条")
        
        if len(execution_times) > 1:
            print(f"\n📊 详细统计:")
            print(f"  时间标准差: {statistics.stdev(execution_times):.3f}秒")
            print(f"  时间中位数: {statistics.median(execution_times):.3f}秒")
            print(f"  AIMessage长度标准差: {statistics.stdev(response_lengths):.1f} 字符")
            print(f"  AIMessage长度中位数: {statistics.median(response_lengths):.1f} 字符")
            print(f"  AIMessage数量标准差: {statistics.stdev(ai_message_counts):.1f} 条")
            print(f"  AIMessage数量中位数: {statistics.median(ai_message_counts):.1f} 条")
        
        # 详细记录
        print(f"\n🔍 详细记录:")
        for i, (t, l, c) in enumerate(zip(execution_times, response_lengths, ai_message_counts), 1):
            print(f"  第{i}次: {t:.3f}秒, {l} 字符 (AIMessage), {c} 条AIMessage")
            
        # 性能分析
        print(f"\n🎯 性能分析:")
        avg_time = statistics.mean(execution_times)
        avg_length = statistics.mean(response_lengths)
        
        if avg_time < 1.0:
            print("🚀 响应速度: 优秀 (< 1秒)")
        elif avg_time < 3.0:
            print("✅ 响应速度: 良好 (< 3秒)")
        elif avg_time < 5.0:
            print("⚠️  响应速度: 一般 (< 5秒)")
        else:
            print("🐌 响应速度: 需要优化 (> 5秒)")
            
        if avg_length < 500:
            print("📝 AIMessage长度: 简洁 (< 500字符)")
        elif avg_length < 1500:
            print("📄 AIMessage长度: 中等 (< 1500字符)")
        elif avg_length < 3000:
            print("📋 AIMessage长度: 详细 (< 3000字符)")
        else:
            print("📚 AIMessage长度: 冗长 (> 3000字符)")
    
    else:
        print("❌ 没有成功的执行记录")


if __name__ == "__main__":
    example_react()