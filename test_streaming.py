#!/usr/bin/env python3
"""
流式输出功能测试脚本

快速测试新添加的流式生成功能是否正常工作
"""

import sys
import time
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


def test_basic_streaming():
    """测试基本流式生成功能"""
    print("测试基本流式生成...")
    
    try:
        llm = LLM("../models/Qwen3-0.6B")
        
        prompt = "请简单介绍一下Python编程语言。"
        print(f"提示: {prompt}")
        print("生成: ", end='', flush=True)
        
        token_count = 0
        start_time = time.time()
        
        for chunk in llm.stream(prompt, temperature=0.8, max_tokens=50):
            if not chunk['finished']:
                print(chunk['text'], end='', flush=True)
                token_count += 1
            else:
                end_time = time.time()
                duration = end_time - start_time
                speed = token_count / duration if duration > 0 else 0
                print(f"\n✅ 基本流式生成测试通过")
                print(f"   生成了 {token_count} 个token，速度: {speed:.1f} tok/s")
                return True
                
    except Exception as e:
        print(f"\n❌ 基本流式生成测试失败: {e}")
        return False


def test_chat_streaming():
    """测试聊天流式生成功能"""
    print("\n测试聊天流式生成...")
    
    try:
        llm = LLM("../models/Qwen3-0.6B")
        
        prompt = "什么是机器学习？"
        print(f"用户问题: {prompt}")
        print("AI回答: ", end='', flush=True)
        
        response = llm.chat_stream(prompt, temperature=0.7, max_tokens=60)
        
        print(f"✅ 聊天流式生成测试通过")
        print(f"   完整回答长度: {len(response)} 字符")
        return True
        
    except Exception as e:
        print(f"\n❌ 聊天流式生成测试失败: {e}")
        return False


def test_batch_streaming():
    """测试批量流式生成功能"""
    print("\n测试批量流式生成...")
    
    try:
        llm = LLM("../models/Qwen3-0.6B")
        
        prompts = [
            "今天天气如何？",
            "推荐一本书。"
        ]
        
        print("批量提示:")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p}")
        
        print("\n批量生成结果:")
        
        seq_responses = {}
        for batch_output in llm.stream_batch(prompts, temperature=0.8, max_tokens=30):
            for seq_id, chunk in batch_output.items():
                if seq_id not in seq_responses:
                    seq_responses[seq_id] = ""
                    print(f"序列{seq_id}: ", end='', flush=True)
                
                if not chunk['finished']:
                    print(chunk['text'], end='', flush=True)
                    seq_responses[seq_id] += chunk['text']
                else:
                    print(f" [完成]")
        
        print(f"✅ 批量流式生成测试通过")
        print(f"   处理了 {len(seq_responses)} 个序列")
        return True
        
    except Exception as e:
        print(f"\n❌ 批量流式生成测试失败: {e}")
        return False


def test_low_level_api():
    """测试底层API"""
    print("\n测试底层流式API...")
    
    try:
        llm = LLM("../models/Qwen3-0.6B")
        
        sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=40,
            ignore_eos=False
        )
        
        prompt = "人工智能的定义是什么？"
        print(f"提示: {prompt}")
        print("生成: ", end='', flush=True)
        
        for chunk in llm.generate_stream(prompt, sampling_params):
            if not chunk['finished']:
                print(chunk['text'], end='', flush=True)
            else:
                print(f"\n✅ 底层API测试通过")
                print(f"   最终文本长度: {len(chunk['total_text'])} 字符")
                print(f"   Token数量: {len(chunk['total_tokens'])}")
                return True
                
    except Exception as e:
        print(f"\n❌ 底层API测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("nanovllm 流式输出功能测试")
    print("=" * 60)
    
    tests = [
        test_basic_streaming,
        test_chat_streaming,
        test_batch_streaming,
        test_low_level_api,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            break
        except Exception as e:
            print(f"测试执行出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有流式输出功能测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查实现")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 