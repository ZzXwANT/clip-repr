from model.ImageEncoder import VIT, PatchEmbedding
from model.TextEncoder import TextToeknizer, TextEmbedding, TextEncoder
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

img_size=224
patch_size=32 
in_chans=3
vocab_size=49152 + 256
context_length=77
embed_dims=768
num_heads=12
hidden_dims=2048
num_layers=12
output_dims=512
 
image = Image.open('/Users/mac/proj/clip-repr/docs/image_encoder.png').convert('RGB')
text = "A photo of image_encoder by clip"

img_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dims)
tokenizer = TextToeknizer(vocab_size, merges_path="/Users/mac/proj/clip-repr/model/bpe_simple_vocab_16e6.txt")
text_embed = TextEmbedding(vocab_size, embed_dims)


# 图像处理
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
img = transform(image).unsqueeze(0) # (1, 3, 244, 244)

# 文本处理
input_text = '<|startoftext|> ' + text + ' <|endoftext|>'
bpe_text, tokens = tokenizer.encode(input_text)   # eg: [424, 23, 124, ...]
eot_token = tokens[-1]

if len(tokens) > context_length:
    tokens = tokens[:context_length]
    tokens[-1] = eot_token
    
eot_idx = len(tokens) - 1  # 截断后、padding 前记录
tokens += [0] * (context_length - len(tokens))

tokens = torch.tensor(tokens).unsqueeze(0)  # [bz, context_length]
text_embeddings = text_embed(tokens)        # [bz, context_length, embed_dims]

# encoder
img_encoder = VIT(img_size, patch_size, in_chans, embed_dims, num_heads, hidden_dims, num_layers, output_dims)  # (bz, CLS(1), output_dims)
text_encoder =  TextEncoder(embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length)  # (bz, context_length, output_dims)

img_features = img_encoder(img)
text_features = text_encoder(text_embeddings)[:, eot_idx, :]

print("")