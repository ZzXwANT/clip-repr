import torch.nn as nn
import torch
from .LayerNorm import LayerNorm    
from .Transformer import Transformer
    
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dims):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.embedding = nn.Parameter(
            torch.empty(self.vocab_size, self.embedding_dims)
        )
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)
        
    def forward(self, tokens):
        return self.embedding[tokens]   # 查表

class TextEncoder(nn.Module):
    def __init__(self, embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length, attn_mask):
        super().__init__()
        self.ln_final = LayerNorm(embed_dims)
        self.text_proj = nn.Linear(embed_dims, output_dims, bias=False) # 统一映射output_dims
        self.pos_embedding = nn.Parameter(
            torch.randn(1, context_length, embed_dims)
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.01) # 添加 std=0.01 
        self.transformer = Transformer(embed_dims, num_heads, hidden_dims, num_layers, attn_mask, output_dims)
        
    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        x = self.text_proj(x)
        
        return x
if __name__ == "__main__":
    TextEmbedding = TextEmbedding(8, 768)
    embeddings = TextEmbedding(torch.randint(8,(1, 8)))
    
    
    