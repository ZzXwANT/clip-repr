import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, embed_dims, eps=1e-5):
        super().__init__()
        self.embed_dims = embed_dims
        self.gamma = nn.Parameter(torch.ones(self.embed_dims))  # scale
        self.beta = nn.Parameter(torch.zeros(self.embed_dims))  # bias
        self.eps = eps
    
    # 使用 F.layer_norm 加速
    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=(self.embed_dims,),
            weight=self.gamma,
            bias=self.beta,
            eps=self.eps,
        )
    
    def _forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                   # (B, N, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)     # (B, N, 1) 无偏方差
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * normalized + self.beta

        # 查看bf16是否使用
        # if not getattr(self, "_dtype_printed", False):
        #     print(
        #         f"LayerNorm: input={x.dtype}, "
        #         f"mean={mean.dtype}, var={var.dtype}, "
        #         f"normalized={normalized.dtype}, "
        #         f"gamma={self.gamma.dtype}, output={out.dtype}"
        #     )
        #     self._dtype_printed = True
            
        return out   # (B, N, D)
    
if __name__ == "__main__":
    ln = LayerNorm(embed_dims=768)
    x = torch.rand(2, 10, 768)   # (B, N, D)
    out = ln(x)
    print(out.shape)   # (2, 10, 768)