import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dims=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.n_patches = (self.img_size // self.patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels = self.in_chans,
            out_channels = self.embed_dims,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            )
        
    def forward(self, x):     # (B, C ,img_size ,img_size)
        x = self.proj(x)      # (B, embed,  n_patches^1/2 ,n_patches^1/2)                
        x = torch.flatten(x, 2)         # (B, embed ,n_patches)
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

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    patch_embedding = PatchEmbedding(224, 32, 3, 768)
    x = patch_embedding(x)
    print(x.shape)