import torch
import torch.nn as nn
from .MHA import MultiHeadAtten 
from .LayerNorm import LayerNorm
from .GELU import GELU

class ResAttenBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, hidden_dims, mask):
        super().__init__()
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims  
        self.num_heads = num_heads
        self.mask = mask
        self.atten = MultiHeadAtten(self.embed_dims, self.num_heads, self.mask)
        self.ln1 = LayerNorm(self.embed_dims)
        self.ln2 = LayerNorm(self.embed_dims)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.hidden_dims),
            GELU(),
            nn.Linear(self.hidden_dims, self.embed_dims)
        )
        
    def forward(self, x):
        x = x + self.atten(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    model = ResAttenBlock(embed_dims=8, num_heads=2, hidden_dims=16)
    out = model(x)
    print(out.shape)