from bytes_to_unicode import bytes_to_unicode
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
    def __init__(self, merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        self.byte_encoder = bytes_to_unicode()  # dict(int, str) 
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # dict(str, int) 
        
        # 构造 vocab
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + '</w>' for v in vocab]  # 保证每个字符都能被编码
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        
    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]
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
            
            for i in idxs:
                # 有问题！ 
                new_tokens = tokens[:i] + [''.join(best_pair)] + tokens[i+2:]
            tokens = new_tokens
            
            if len(tokens) == 1:
                return tokens
    
    def encode(self, text):
        # gpt-2 预处理
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for word in re.findall(self.pat, text):
            word_unic = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
            bpe_tokens.extend(self.encoder[token] for token in self.bpe(word_unic))
        return bpe_tokens
    
    def decode(self, tokens):
        word_unic = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in word_unic]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        return text
    
if __name__ == "__main__":
    tokenizer = TextToeknizer(merges_path="/Users/mac/proj/clip-repr/model/bpe_simple_vocab_16e6.txt")
    tokenizer.decode([585, 533, 13306])
    # tokenizer.encode("it is unbelievable")
    
    