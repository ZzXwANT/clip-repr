import torch
import torch.nn as nn
from .MHA import MultiHeadAtten 
from .LayerNorm import LayerNorm
from .GELU import GELU

class ResAttenBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, hidden_dims, mask, dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims  
        self.num_heads = num_heads
        self.mask = mask
        self.atten = MultiHeadAtten(self.embed_dims, self.num_heads, self.mask)
        self.attn_dropout = nn.Dropout(dropout)

        self.ln1 = LayerNorm(self.embed_dims)
        self.ln2 = LayerNorm(self.embed_dims)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.hidden_dims),
            # 保留线性层索引 0、2，兼容旧 checkpoint 和文本塔初始化。
            nn.Sequential(GELU(), nn.Dropout(dropout)),
            nn.Linear(self.hidden_dims, self.embed_dims),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn_dropout(self.atten(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    model = ResAttenBlock(embed_dims=8, num_heads=2, hidden_dims=16, mask=True)
    out = model(x)
    print(out.shape)
