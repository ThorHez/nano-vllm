from typing import Iterator, Dict, Any, List, Optional, Callable
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class LLM(LLMEngine):
    """
    高级LLM接口，提供便捷的生成方法
    """
    
    def stream(
        self,
        prompt: str | list[int],
        temperature: float = 1.0,
        max_tokens: int = 64,
        ignore_eos: bool = False,
        use_tqdm: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        便捷的单个提示流式生成方法
        
        Args:
            prompt: 输入提示
            temperature: 采样温度
            max_tokens: 最大生成token数
            ignore_eos: 是否忽略结束符
            use_tqdm: 是否显示进度条
            tools: 可用工具列表
            tool_choice: 工具选择策略
            
        Yields:
            生成的增量输出字典
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            tools=tools,
            tool_choice=tool_choice
        )
        
        yield from self.generate_stream(prompt, sampling_params, use_tqdm)
    
    def stream_batch(
        self,
        prompts: list[str] | list[list[int]],
        temperature: float = 1.0,
        max_tokens: int = 64,
        ignore_eos: bool = False,
        use_tqdm: bool = False,
    ) -> Iterator[Dict[int, Dict[str, Any]]]:
        """
        便捷的批量流式生成方法
        
        Args:
            prompts: 输入提示列表
            temperature: 采样温度
            max_tokens: 最大生成token数
            ignore_eos: 是否忽略结束符
            use_tqdm: 是否显示进度条
            tools: 可用工具列表
            tool_choice: 工具选择策略
            
        Yields:
            批量生成的增量输出字典
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            tools=tools,
            tool_choice=tool_choice
        )
        
        yield from self.generate_stream_batch(prompts, sampling_params, use_tqdm)

