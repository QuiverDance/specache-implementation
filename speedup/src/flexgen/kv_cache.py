import torch

class SimpleKV:
    """
    GPU-resident 16-bit KV Cache.
    heads × max_len × head_dim
    """
    def __init__(self, heads: int, dim: int, max_len: int, device):
        self.K = torch.empty(heads, max_len, dim, dtype=torch.float16, device=device)
        self.V = torch.empty_like(self.K)
        self.cur = 0                    

    def append(self, k: torch.Tensor, v: torch.Tensor):
        # k,v : (B, H, 1, D) -B==1 in FlexGen
        idx = self.cur
        self.K[:, idx] = k[0]
        self.V[:, idx] = v[0]
        self.cur += 1

    def slice(self):
        return self.K[:, :self.cur], self.V[:, :self.cur]  # (H, L, D)
