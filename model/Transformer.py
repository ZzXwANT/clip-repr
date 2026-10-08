import torch
import torch.nn as nn
from ResAttenBlock import ResAttenBlock

class Transformer(nn.Module):
    def __init__(self, embed_dims, num_heads, hidden_dims, num_layers, mask):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.mask = mask
        self.layes = nn.ModuleList(
            ResAttenBlock(embed_dims=self.embed_dims, num_heads=self.num_heads, hidden_dims=self.hidden_dims, mask = self.mask)
            for _ in range(self.num_layers)         
        )
        
    def forward(self, x): 
        for layer in self.layes:
            x = layer(x)
        return x

if __name__ == "__main__":
    T = Transformer(embed_dims=8, num_heads=2, hidden_dims=16, num_layers=3)
    print(T.layes)
