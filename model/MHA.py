import torch
import torch.nn as nn

class MultiHeadAtten(nn.Module):
    def __init__(self, embed_dims, num_heads, mask=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mask = mask
        self.head_dims = self.embed_dims // self.num_heads
        self.scale = self.head_dims ** 0.5
        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3)  # （D, 3*D）
        
        # self.q = nn.ModuleList([
        #     nn.Linear(self.embed_dims, self.head_dims)
        #     for _ in range(self.num_heads)
        # ])
        # self.k = nn.ModuleList([
        #     nn.Linear(self.embed_dims, self.head_dims)
        #     for _ in range(self.num_heads)
        # ])
        # self.v = nn.ModuleList([
        #     nn.Linear(self.embed_dims, self.head_dims)
        #     for _ in range(self.num_heads)
        # ])  
        
        self.o_proj = nn.Linear(self.embed_dims, self.embed_dims)   # （D, D）
        
    def forward(self, x):
        B, N, D = x.shape                  # D = embed_dims
        # (B, N, 3, num_heads, head_dims)   分多头 atten
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims)  
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, num_heads, N, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, num_heads, N, head_dims)
        
        # (B, num_heads, N, N)
        qkT = (q @ k.transpose(-2, -1)) / self.scale
        
        # 因果注意力掩码
        if self.mask is not None:
            # 上三角 -inf 掩码
            casusal_mask = torch.triu(torch.full((N,N), float("-inf")), diagonal=1)   
            
            # casusal_mask = torch.zeros(N,N)
            # for i in range(N):
            #     for j in range(N):
            #         if j > i:
            #             casusal_mask[i][j] = float("-inf")
                        
            attn = torch.softmax(qkT + casusal_mask, dim=-1)
        else:
            attn = torch.softmax(qkT, dim=-1)
        
        # (B, num_heads, N, head_dims)
        score = attn @ v
        # (B, N, embed_dims or D)
        socre = torch.transpose(score, 1, 2).reshape(B, N, D)   
        
        # (num_heads, B, N, head_dims)
        # q = torch.stack([self.q[i](x) for i in range(self.num_heads)]) 
        # k = torch.stack([self.k[i](x) for i in range(self.num_heads)])
        # v = torch.stack([self.v[i](x) for i in range(self.num_heads)])
        
        # (num_heads, B, N, head_dims)
        # socre = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1) @ v 
        # (B, N, embed_dims or D)  
        # socre = torch.permute(socre, (1, 2, 0, 3)).reshape(B, N, D)   
        
        return self.o_proj(socre)   # 混合多头 atten
    
if __name__ == "__main__":
    mha = MultiHeadAtten(embed_dims=768, num_heads=12)
    x = torch.rand(2, 10, 768)   # (B, N, D)
    out = mha(x)
    print(out.shape)   # (2, 10, 768)