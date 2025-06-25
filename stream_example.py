#!/usr/bin/env python3
"""
流式输出示例

展示如何使用nanovllm的流式生成功能
"""

import sys
import time
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


def single_stream_example():
    """
    单个提示的流式生成示例
    """
    print("=== 单个提示流式生成示例 ===")
    
    # 初始化LLM
    llm = LLM("/root/models/Qwen3-0.6B")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        ignore_eos=False
    )
    
    prompt = "人工智能的未来发展方向是什么？请详细分析。"
    print(f"提示: {prompt}")
    print("生成中...")
    print("-" * 50)
    
    # 流式生成
    for chunk in llm.generate_stream(prompt, sampling_params, use_tqdm=True):
        if not chunk['finished']:
            # 实时输出新生成的文本片段
            print(chunk['text'], end='', flush=True)
        else:
            # 生成完成
            print("\n" + "-" * 50)
            print("生成完成!")
            print(f"完整回答: {chunk['total_text']}")
            print(f"总token数: {len(chunk['total_tokens'])}")


def batch_stream_example():
    """
    批量流式生成示例
    """
    print("\n=== 批量流式生成示例 ===")
    
    # 初始化LLM
    llm = LLM("/root/models/Qwen3-0.6B")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        ignore_eos=False
    )
    
    prompts = [
        "今天天气怎么样？",
        "Python编程的优势是什么？",
        "推荐一道简单的菜谱。"
    ]
    
    print("提示列表:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    print("\n生成中...")
    print("-" * 50)
    
    # 跟踪每个序列的输出
    seq_outputs = {}
    
    # 批量流式生成
    for batch_output in llm.generate_stream_batch(prompts, sampling_params, use_tqdm=True):
        for seq_id, chunk in batch_output.items():
            if seq_id not in seq_outputs:
                seq_outputs[seq_id] = {
                    'text': '',
                    'finished': False
                }
            
            if not chunk['finished']:
                # 累积文本
                seq_outputs[seq_id]['text'] += chunk['text']
                print(f"序列{seq_id}: {chunk['text']}", end='', flush=True)
            else:
                # 序列完成
                seq_outputs[seq_id]['finished'] = True
                print(f"\n序列{seq_id}完成: {chunk['total_text']}")
                print("-" * 30)
    
    print("所有序列生成完成!")


def interactive_stream_example():
    """
    交互式流式生成示例
    """
    print("\n=== 交互式流式生成示例 ===")
    print("输入'quit'退出")
    
    # 初始化LLM
    llm = LLM("/root/models/Qwen3-0.6B")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=150,
        ignore_eos=False
    )
    
    while True:
        try:
            prompt = input("\n请输入提示: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                continue
            
            print("回答: ", end='', flush=True)
            start_time = time.time()
            token_count = 0
            
            # 流式生成
            for chunk in llm.generate_stream(prompt, sampling_params):
                if not chunk['finished']:
                    print(chunk['text'], end='', flush=True)
                    token_count += 1
                else:
                    end_time = time.time()
                    duration = end_time - start_time
                    tokens_per_second = token_count / duration if duration > 0 else 0
                    
                    print(f"\n[生成完成 - {token_count} tokens, {tokens_per_second:.1f} tok/s]")
                    
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


def main():
    """
    主函数
    """
    print("nanovllm 流式输出示例")
    print("=" * 60)
    
    # 运行单个流式生成示例
    single_stream_example()
    
    # 运行批量流式生成示例
    batch_stream_example()
    
    # 运行交互式流式生成示例
    interactive_stream_example()

if __name__ == "__main__":
    main() 