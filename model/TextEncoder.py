from .bytes_to_unicode import bytes_to_unicode
import regex as re
import ftfy
import html
import torch.nn as nn
import torch
from .LayerNorm import LayerNorm    
from .Transformer import Transformer

def record_pairs(word):
    pairs_with_idx = dict()
    for idx in range(len(word) - 1):
        pair = (word[idx], word[idx + 1])
        pairs_with_idx.setdefault(pair, []).append(idx) # 防止重复pair覆盖
    return pairs_with_idx

def basic_clean(text):
    text = ftfy.fix_text(text)  # 修复乱码
    text = html.unescape(html.unescape(text))   # 转译html字符
    return text.strip()         # 移除开头结尾空白字符

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)    # 用空格' '替换空白字符
    return text.strip()

class TextToeknizer():
    def __init__(self,vocab_size, merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = merges[1:vocab_size-256-2+1 - 256]        # 多 *</w>
        merges = [tuple(merge.split()) for merge in merges]
        self.byte_encoder = bytes_to_unicode()  # dict(int, str) 
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # dict(str, int) 
        
        # 构造 vocab
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + '</w>' for v in vocab]  # 256
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 2
        self.encoder = dict(zip(vocab, range(len(vocab))))
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        
    def bpe(self, word):
        if word in self.cache:
            return [self.cache[word]]
        tokens = list(word[:-1]) + [word[-1] + '</w>']        
        
        while True:
            pairs_with_idx = record_pairs(tokens) # {'pairs': idx}
            candidates = [
                pair for pair in pairs_with_idx
                if pair in self.bpe_ranks
            ]
            if not candidates:
                return tokens
            
            best_pair = min(candidates, key=lambda pair: self.bpe_ranks[pair])
            idxs = pairs_with_idx[best_pair]
            
            new_tokens = []
            i = 0 
            while i < len(tokens):
                if i in idxs:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
                    
            tokens = new_tokens
            
            if len(tokens) == 1:
                return tokens
    
    def encode(self, text):
        # gpt-2 预处理
        bpe_text = []
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for word in re.findall(self.pat, text):
            word_unic = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
            for token in self.bpe(word_unic):
                bpe_text.append(token)
                bpe_tokens.append(self.encoder[token])
            
        return bpe_text, bpe_tokens
    
    def decode(self, tokens):
        word_unic = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in word_unic]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        return text
    
    
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
    def __init__(self, embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length):
        super().__init__()
        self.ln_final = LayerNorm(embed_dims)
        self.text_proj = nn.Linear(embed_dims, output_dims, bias=False)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, context_length, embed_dims)
        )
        self.transformer = Transformer(embed_dims, num_heads, hidden_dims, num_layers, True, output_dims)
        
    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        x = self.text_proj(x)
        
        return x
if __name__ == "__main__":
    TextEmbedding = TextEmbedding(8, 768)
    embeddings = TextEmbedding(torch.randint(8,(1, 8)))
    
    tokenizer = TextToeknizer(merges_path="/Users/mac/proj/clip-repr/model/bpe_simple_vocab_16e6.txt")
    # tokenizer.decode([585, 533, 13306])
    tokenizer.encode("aaaa it is unbelievable")
    
    
    