import torch
import torch.nn as nn
from LayerNorm import LayerNorm
from Transformer import Transformer
from Patch_Embeding import PatchEmbedding

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
        x = self.ln_pre(x)
        x = self.Transformer(x)
        x = self.ln_post(x[:, 0, :])  # 取出cls_token对应的输出
        x = self.o_proj(x)
        
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = VIT(img_size=224, patch_size=32, in_chans=3, embed_dims=8, num_heads=2, hidden_dims=16, num_layers=3, output_dims=10)
    out = model(x)
    print(out.shape)  # (2, 10)