import torch
import torch.nn as nn
from .LayerNorm import LayerNorm
from .Transformer import Transformer

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dims=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.n_patches = (self.img_size // self.patch_size) ** 2
        # 分块 + 线性映射
        self.proj = nn.Conv2d(
            in_channels = self.in_chans,
            out_channels = self.embed_dims,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            )
        
    def forward(self, x):     # (B, C, img_size, img_size)
        x = self.proj(x)      # (B, embed,  n_patches^1/2, n_patches^1/2)                
        x = torch.flatten(x, 2)         # (B, embed, n_patches)
        x = torch.transpose(x, 1, 2)    # (B, n_patch, embed)
        return x
        
        # 原始算法
        B, C, H, W = x.shape
        patches = (
            x.reshape(
                B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size
                )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(
                B, H//self.patch_size*H//self.patch_size, C*self.patch_size*self.patch_size
                )
            )
        linear = nn.Linear(C * self.patch_size * self.patch_size, self.embed_dim)
        return linear(patches)
    
class VIT(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dims, num_heads, hidden_dims, num_layers, output_dims):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dims))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (img_size//patch_size)**2 + 1, embed_dims)
        )
        
        self.ln_pre = LayerNorm(embed_dims)
        self.ln_post = LayerNorm(embed_dims)
        
        self.PatchEmbedding = PatchEmbedding(img_size, patch_size, in_chans, embed_dims)
        self.Transformer = Transformer(embed_dims, num_heads, hidden_dims, num_layers)
        self.o_proj = nn.Linear(embed_dims, output_dims, bias=False)
        
    def forward(self,x):
        x = self.PatchEmbedding(x)  # (B, n_patches, embed_dims)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = x + self.pos_embedding
        x = self.ln_pre(x)            # 论文中额外的ln
        x = self.Transformer(x)
        x = self.ln_post(x[:, 0, :])  # 取出cls_token对应的输出
        x = self.o_proj(x)
        
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = VIT(img_size=224, patch_size=32, in_chans=3, embed_dims=8, num_heads=2, hidden_dims=16, num_layers=3, output_dims=10)
    out = model(x)
    print(out.shape)  # (2, 10)

    x = torch.randn(2, 3, 224, 224)
    patch_embedding = PatchEmbedding(224, 32, 3, 768)
    x = patch_embedding(x)
    print(x.shape)  # (2, 49, 768)  