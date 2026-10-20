from .bytes_to_unicode import bytes_to_unicode
from collections import OrderedDict
import regex as re
import ftfy
import html


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
    def __init__(self,vocab_size, merges_path, cache_size=32768):
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self.cache_size = cache_size
        self._bpe_cache = OrderedDict()
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
        if word in self._bpe_cache:
            self._bpe_cache.move_to_end(word)
            # Keep cached pieces immutable; callers may modify returned lists.
            return list(self._bpe_cache[word])
        tokens = list(word[:-1]) + [word[-1] + '</w>']        
        
        while True:
            pairs_with_idx = record_pairs(tokens) # {'pairs': idx}
            candidates = [
                pair for pair in pairs_with_idx
                if pair in self.bpe_ranks
            ]
            if not candidates:
                break
            
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
                break
        if self.cache_size:
            self._bpe_cache[word] = tuple(tokens)
            if len(self._bpe_cache) > self.cache_size:
                self._bpe_cache.popitem(last=False)
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

    def encode_batch(self, texts):
        """Encode captions in order, encoding identical captions only once per batch."""
        if isinstance(texts, str):
            raise TypeError("encode_batch expects an iterable of strings, not one string")
        unique = {}
        result = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("each caption must be a string")
            if text not in unique:
                unique[text] = self.encode(text)
            pieces, ids = unique[text]
            result.append((pieces.copy(), ids.copy()))
        return result

    def tokenize_batch(self, texts, context_length=77):
        """Return [batch, context_length] int64 tokens, including SOT/EOT and padding.

        Supports 512 captions or any other batch size. BPE rules and IDs are
        unchanged, so existing model checkpoints remain compatible.
        """
        import torch

        if not isinstance(context_length, int) or context_length < 2:
            raise ValueError("context_length must be an integer >= 2")
        sot = self.encoder['<|startoftext|>']
        eot = self.encoder['<|endoftext|>']
        rows = []
        for _, ids in self.encode_batch(texts):
            ids = [i for i in ids if i not in (sot, eot)][:context_length - 2]
            row = [sot, *ids, eot]
            rows.append(row + [0] * (context_length - len(row)))
        # One tensor construction for the entire batch, including empty batches.
        return torch.tensor(rows, dtype=torch.long).reshape(len(rows), context_length)
    
    def decode(self, tokens):
        word_unic = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in word_unic]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        return text

if __name__ == "__main__":
    from pathlib import Path
    from config import config

    tokenizer = TextToeknizer(
        vocab_size=config.vocab_size,
        merges_path=Path(__file__).resolve().parents[1] / "model" / "bpe_simple_vocab_16e6.txt",
    )
    # tokenizer.decode([585, 533, 13306])
    tokenizer.encode("aaaa it is unbelievable")
