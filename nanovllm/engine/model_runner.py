import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from line_profiler import LineProfiler

lp = LineProfiler()


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        # 每个KV缓存块的大小，较大的块可以减少缓存切换的次数，但会占用更多的内存，造成显存的浪费
        self.block_size = config.kvcache_block_size
        # 是否强制使用eager模式，如果为True，则不使用CUDAGraph，而是直接执行模型计算
        self.enforce_eager = config.enforce_eager
        # 进行TP时总共使用的GPU数量
        self.world_size = config.tensor_parallel_size
        # TP并行时主进程的rank编号
        self.rank = rank
        # 用于进程间通信的事件对象，例如：在TP并行时，用于call函数调用的通知
        self.event = event

        # 初始化分布式推理，使用NCCL作为通信后端，使用TCP协议进行通信，
        # world_size为TP并行时总共使用的GPU数量，
        # rank为TP并行时主进程的rank编号
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        # 设置当前进程使用的GPU设备
        torch.cuda.set_device(rank)
        # 设置默认的浮点数类型
        default_dtype = torch.get_default_dtype()
        # 设置默认的浮点数类型
        torch.set_default_dtype(hf_config.torch_dtype)
        # 设置默认的设备为GPU
        torch.set_default_device("cuda")
        # 加载模型，加载模型时使用cuda设备加载
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载模型权重
        load_model(self.model, config.model)
        # 初始化采样器
        self.sampler = Sampler()
        # 分配KV缓存
        self.allocate_kv_cache(config.gpu_memory_utilization)
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认的设备和浮点数类型
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 分配共享内存，用于TP并行时进程间通信，主要是用于发送函数调用信息
        if self.world_size > 1:
            if rank == 0:
                # 尝试清理已存在的共享内存
                try:
                    existing_shm = SharedMemory(name="nanovllm")
                    existing_shm.close()
                    existing_shm.unlink()
                except FileNotFoundError:
                    # 共享内存不存在，这是正常情况
                    pass
                except Exception as e:
                    print(f"清理共享内存时出现警告: {e}")

                # 创建新的共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, gpu_memory_utilization):
        config = self.config
        hf_config = config.hf_config
        # 获取当前GPU的空闲内存和总内存
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 计算每个gpu分配多少个注意力头
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 根据注意力计算每个块实际需要用到的显存大小bytes，需要乘以2是因为kv_cache需要q和k两个缓存
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )

        # 计算可用于KV缓存的内存
        available_memory = int(total * gpu_memory_utilization - used)
        config.num_kvcache_blocks = available_memory // block_bytes

        # 检查是否有足够的内存分配KV缓存
        if config.num_kvcache_blocks <= 0:
            # 如果内存不足
            print(f"警告: GPU内存不足，只能分配 {config.num_kvcache_blocks} 个KV缓存块")
            print(f"总内存: {total / 1024**3:.2f}GB, 已使用: {used / 1024**3:.2f}GB")
            print(
                f"可用内存: {available_memory / 1024**3:.2f}GB, 每块需要: {block_bytes / 1024**2:.2f}MB"
            )
            raise ValueError("GPU内存不足，无法分配KV缓存")
        else:
            print(
                f"分配 {config.num_kvcache_blocks} 个KV缓存块 (每块 {block_bytes / 1024**2:.2f}MB)"
            )

        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        layer_id = 0
        # 将kv_cache分配给模型中的每个注意力层
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        assert len(input_ids) == len(slot_mapping)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token) 
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # @lp
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 数据准备，所有进程执行相同的数据准备操作
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )

        # 采样器准备，只有主进程执行采样操作
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 模型推理，所有进程执行相同的推理操作
        logits = self.run_model(input_ids, positions, is_prefill)

        # 采样，只有主进程执行采样操作
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
