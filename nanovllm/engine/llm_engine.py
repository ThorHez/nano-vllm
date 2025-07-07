import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from typing import Iterator, Dict, Any, List, Optional
import hashlib

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.tools.tool_executor import ToolExecutor

from nanovllm.engine.sequence import SequenceStatus


class LLMEngine:
    # 工具调用的特殊token_id
    TOOL_CALL_START_TOKEN = 151657  # <tool_call>
    TOOL_CALL_END_TOKEN = 151658  # </tool_call>
    TOOL_RESPONSE_START_TOKEN = 151665  # <tool_response>
    TOOL_RESPONSE_END_TOKEN = 151666  # </tool_response>
    IM_START_TOKEN = 151644  # <im_start>
    IM_END_TOKEN = 151645  # </im_end>
    TOOL_ROLE_TOKEN = 14172  # tool role
    LINE_BREAK_TOKEN = 198  # line break
    ASSISTANT_TOKEN = 77091  # assistant role
    THINK_START_TOKEN = 151667  # <think>
    USER_TOKEN = 872  # user role

    def __init__(self, model, enable_tools: bool = True, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 启动多进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程需要启动
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # 初始化工具执行器
        self.enable_tools = enable_tools
        self.tool_executor = ToolExecutor() if enable_tools else None

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 进行好处理，判断seq是否生成eos，如果接收到eos标志，就序列停止生成，并清空kv_cache
        self.scheduler.postprocess(seqs, token_ids)

        # 检查是否有序列包含工具调用
        if self.enable_tools and self.tool_executor:
            self._process_tool_calls(seqs)

        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def _process_tool_calls(self, seqs: List[Sequence]):
        """处理序列中的工具调用"""
        # 如果没有启用工具或工具执行器不存在，直接返回
        if not self.enable_tools or not self.tool_executor:
            return

        for seq in seqs:
            if seq.is_finished:
                continue

            # 检查token序列中是否包含工具调用标记
            token_ids = seq.completion_token_ids
            if seq.tool_call_processed_begin == -1:
                has_tool_call_start = self.TOOL_CALL_START_TOKEN in token_ids
                has_tool_call_end = self.TOOL_CALL_END_TOKEN in token_ids
                decode_token_ids = seq.completion_token_ids
            else:
                has_tool_call_start = (
                    self.TOOL_CALL_START_TOKEN
                    in token_ids[seq.tool_call_processed_begin :]
                )
                has_tool_call_end = (
                    self.TOOL_CALL_END_TOKEN
                    in token_ids[seq.tool_call_processed_begin :]
                )
                decode_token_ids = token_ids[seq.tool_call_processed_begin :]

            # 只有当包含完整的工具调用标记时才处理
            if has_tool_call_start and has_tool_call_end:
                # 获取当前生成的文本用于解析
                current_text = self.tokenizer.decode(decode_token_ids)

                # 解析并执行工具调用
                tool_calls, results = self.tool_executor.parse_and_execute(
                    current_text, seq.tool_call_hash_set
                )

                if tool_calls and results:
                    # 格式化工具执行结果，这种场景只可能有一个工具调用的结果
                    tool_results_text = self.tool_executor.format_tool_results(
                        [results[0]]
                    )

                    # 添加工具调用结束标记
                    # seq.append_token(self.IM_END_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # 添加工具调用接口开始标记
                    # seq.append_token(self.IM_START_TOKEN)
                    # seq.append_token(self.TOOL_ROLE_TOKEN)
                    seq.append_token(self.LINE_BREAK_TOKEN)

                    # 添加工具结果开始标记
                    seq.append_token(self.TOOL_RESPONSE_START_TOKEN)
                    seq.append_token(self.LINE_BREAK_TOKEN)

                    # 将工具结果编码并添加到序列中
                    tool_result_tokens = self.tokenizer.encode(
                        tool_results_text, add_special_tokens=False
                    )
                    for token_id in tool_result_tokens:
                        seq.append_token(token_id)

                    # 添加工具结果结束标记
                    seq.append_token(self.LINE_BREAK_TOKEN)
                    seq.append_token(self.TOOL_RESPONSE_END_TOKEN)
                    seq.append_token(self.LINE_BREAK_TOKEN)
                    # seq.append_token(self.IM_END_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # seq.append_token(self.IM_START_TOKEN)
                    # seq.append_token(self.USER_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # 添加用户提示词
                    # user_tokens = self.tokenizer.encode(
                    #     """Answer the question according to the tool response or if you don't have the information,
                    #     you can call the tool again, but you should not call the same tool again""",
                    #     add_special_tokens=False,
                    # )
                    # for token_id in user_tokens:
                    #     seq.append_token(token_id)
                    # seq.append_token(self.IM_END_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # next_token = self.tokenizer.encode(
                    #     "Next, according to the tool response: ",
                    #     add_special_tokens=False,
                    # )
                    # for token_id in next_token:
                    #     seq.append_token(token_id)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # 已添加工具调用结果，下一步继续生成AI的回复
                    # seq.append_token(self.IM_START_TOKEN)
                    # seq.append_token(self.ASSISTANT_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # seq.append_token(self.THINK_START_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                    # 重置序列状态为运行中，确保能继续生成
                    seq.status = SequenceStatus.RUNNING
                else:
                    # 同样的工具已经调用过，加入提示词继续生成
                    seq.append_token(self.LINE_BREAK_TOKEN)
                    seq.append_token(self.IM_START_TOKEN)
                    seq.append_token(self.ASSISTANT_TOKEN)
                    seq.append_token(self.LINE_BREAK_TOKEN)
                    # seq.append_token(self.THINK_START_TOKEN)
                    # seq.append_token(self.LINE_BREAK_TOKEN)

                # 标记该序列已处理工具调用，避免重复处理
                seq.tool_call_processed_begin = len(seq.completion_token_ids)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        use_tqdm: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式生成单个提示的响应

        Args:
            prompt: 输入提示，可以是字符串或token列表
            sampling_params: 采样参数
            use_tqdm: 是否显示进度条

        Yields:
            包含增量输出的字典，格式为:
            {
                'text': str,  # 新生成的文本片段
                'token_id': int,  # 新生成的token ID
                'finished': bool,  # 是否已完成生成
                'total_text': str,  # 到目前为止生成的全部文本
                'total_tokens': list[int],  # 到目前为止生成的全部token
            }
        """
        if use_tqdm:
            pbar = tqdm(desc="生成中", dynamic_ncols=True)

        # 添加请求到调度器
        self.add_request(prompt, sampling_params)

        # 获取序列ID（假设只有一个序列）
        target_seq_id = None
        for seq in self.scheduler.waiting:
            target_seq_id = seq.seq_id
            break

        if target_seq_id is None:
            for seq in self.scheduler.running:
                target_seq_id = seq.seq_id
                break

        previous_tokens = []
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            outputs, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "prefill": f"{int(prefill_throughput)}tok/s",
                        "decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            # 查找目标序列的新token
            for seq in self.scheduler.running:
                if seq.seq_id == target_seq_id:
                    current_tokens = seq.completion_token_ids
                    if len(current_tokens) > len(previous_tokens):
                        # 有新的token生成
                        new_tokens = current_tokens[len(previous_tokens) :]
                        for new_token_id in new_tokens:
                            new_text = self.tokenizer.decode([new_token_id])
                            total_text = self.tokenizer.decode(current_tokens)

                            yield {
                                "text": new_text,
                                "token_id": new_token_id,
                                "finished": False,
                                "total_text": total_text,
                                "total_tokens": current_tokens.copy(),
                            }

                            if use_tqdm:
                                pbar.update(1)

                        previous_tokens = current_tokens.copy()
                    break

            # 检查序列是否完成
            for seq_id, token_ids in outputs:
                if seq_id == target_seq_id:
                    final_text = self.tokenizer.decode(token_ids)

                    yield {
                        "text": "",
                        "token_id": None,
                        "finished": True,
                        "total_text": final_text,
                        "total_tokens": token_ids,
                    }
                    break

        if use_tqdm:
            pbar.close()

    def generate_stream_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> Iterator[Dict[int, Dict[str, Any]]]:
        """
        批量流式生成多个提示的响应

        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数或采样参数列表
            use_tqdm: 是否显示进度条

        Yields:
            字典，键为序列ID，值为增量输出字典
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="批量生成中", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求
        seq_id_to_index = {}
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            seq = Sequence(prompt, sp)
            self.scheduler.add(seq)
            seq_id_to_index[seq.seq_id] = i

        previous_tokens = {seq_id: [] for seq_id in seq_id_to_index.keys()}
        completed_seqs = set()
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            outputs, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "prefill": f"{int(prefill_throughput)}tok/s",
                        "decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            step_output = {}

            # 处理运行中的序列
            for seq in self.scheduler.running:
                if seq.seq_id in seq_id_to_index and seq.seq_id not in completed_seqs:
                    current_tokens = seq.completion_token_ids
                    prev_tokens = previous_tokens[seq.seq_id]

                    if len(current_tokens) > len(prev_tokens):
                        # 有新的token生成
                        new_tokens = current_tokens[len(prev_tokens) :]
                        for new_token_id in new_tokens:
                            new_text = self.tokenizer.decode([new_token_id])
                            total_text = self.tokenizer.decode(current_tokens)

                            step_output[seq.seq_id] = {
                                "text": new_text,
                                "token_id": new_token_id,
                                "finished": False,
                                "total_text": total_text,
                                "total_tokens": current_tokens.copy(),
                            }

                        previous_tokens[seq.seq_id] = current_tokens.copy()

            # 处理完成的序列
            for seq_id, token_ids in outputs:
                if seq_id in seq_id_to_index:
                    final_text = self.tokenizer.decode(token_ids)
                    seq_sp = sampling_params[seq_id_to_index[seq_id]]

                    step_output[seq_id] = {
                        "text": "",
                        "token_id": None,
                        "finished": True,
                        "total_text": final_text,
                        "total_tokens": token_ids,
                    }
                    completed_seqs.add(seq_id)
                    if use_tqdm:
                        pbar.update(1)

            if step_output:
                yield step_output

        if use_tqdm:
            pbar.close()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[Dict[str, Any]]:
        """
        生成响应，返回包含工具调用信息的结果
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 处理输出，添加工具调用信息
        results = []
        for seq_id in sorted(outputs):
            token_ids = outputs[seq_id]
            text = self.tokenizer.decode(token_ids)

            results.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                }
            )

        if use_tqdm:
            pbar.close()
        return results
