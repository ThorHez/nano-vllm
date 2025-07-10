from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        # 用于存放等待prefill的序列
        self.waiting: deque[Sequence] = deque()
        # 用于存放等待等待decode的序列
        self.running: deque[Sequence] = deque()
        
        self.batch_decode_seqs: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running and not self.batch_decode_seqs
        # return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        
    
    def add_batch_decode_seq(self, seq: Sequence):
        self.batch_decode_seqs.append(seq)
        
        
    def remove_from_running(self, seq: Sequence):
        self.running.remove(seq)
    

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        # 限制单次推理的序列数量，避免GPU显存不足
        num_seqs = 0
        num_batched_tokens = 0
        
        
        while self.batch_decode_seqs and num_seqs < self.max_num_seqs:
            seq = self.batch_decode_seqs[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # 实际需要计算的seq - 已经计算的序列（存放在kv_cache中）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.batch_decode_seqs.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
        
        
        
        # 一次取出足够多的序列用于批量推理
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # 实际需要计算的seq - 已经计算的序列（存放在kv_cache中）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 如果block_manager不能分配额外的cache空间，那么将序列塞回等待的状态，同时删除掉相应的kv_cache，直到可以分配cache空间
            # 给将要decode的seq
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 在running状态的序列优先执行，这样能有更低的时延
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
