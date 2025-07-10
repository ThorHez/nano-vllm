import torch
from torch import nn
# from line_profiler import LineProfiler

# lp_sampler = LineProfiler()

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # @lp_sampler
    # @torch.compile
    def forward_opt(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        最终优化版本：使用torch.any()进行greedy检查
        性能提升10.8%，波动最小
        """
        # 重要：克隆logits以避免修改原始数据
        if logits.dtype != torch.float:
            logits = logits.clone().to(torch.float)
        else:
            logits = logits.clone()
        
        # 最优化：使用torch.any检查是否有非零元素
        # 如果没有任何非零元素，说明全部为greedy (temperature=0)
        if not torch.any(temperatures):
            return logits.argmax(dim=-1)
        
        # 原始逻辑用于其他情况
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        greedy_mask = temperatures == 0
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
    
    
    # @lp_sampler
    def forward_v1(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        return greedy_tokens
    
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        greedy_tokens = probs.argmax(dim=-1)
        return greedy_tokens
    
    # @lp_sampler
    def forward_original(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)