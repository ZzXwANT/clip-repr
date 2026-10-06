import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embed_dims, eps=1e-5):
        super().__init__()
        self.embed_dims = embed_dims
        self.gamma = nn.Parameter(torch.ones(self.embed_dims))  # scale
        self.beta = nn.Parameter(torch.zeros(self.embed_dims))  # bias
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                   # (B, N, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)     # (B, N, 1) 无偏方差
        ln = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        
        return ln   # (B, N, D)
    
if __name__ == "__main__":
    ln = LayerNorm(embed_dims=768)
    x = torch.rand(2, 10, 768)   # (B, N, D)
    out = ln(x)
    print(out.shape)   # (2, 10, 768)