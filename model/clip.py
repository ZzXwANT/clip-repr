from model.TextEncoder import TextEncoder, TextEmbedding
from model.ImageEncoder import VIT
import torch.nn as nn
import torch
import numpy as np


class clip(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dims_i, embed_dims_t,
                 num_heads_i, num_heads_t, hidden_dims_i, hidden_dims_t,
                 num_layers, output_dims, context_length, vocab_size):
        super().__init__()
        # encoder
        self.img_encoder = VIT(img_size, patch_size, in_chans, embed_dims_i, num_heads_i, hidden_dims_i, num_layers, output_dims)
        self.text_encoder = TextEncoder(embed_dims_t, num_heads_t, hidden_dims_t, num_layers, output_dims, context_length, attn_mask=True)
        
        self.text_embed = TextEmbedding(vocab_size, embed_dims_t)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 原文 1/0.07 初始化 tau
        
    def encode_image(self, imgs):
        features = self.img_encoder(imgs)
        return self.l2_norm(features)

    def encode_text(self, tokens):
        eot_idx = tokens.argmax(dim=-1)
        text_embeddings = self.text_embed(tokens)
        text_features = self.text_encoder(text_embeddings, eot_idx)
        return self.l2_norm(text_features)
    
    def forward(self, imgs, tokens, labels=None, class_tokens=None):
        # (B, output_dims)
        img_features = self.encode_image(imgs)
        if labels is None and class_tokens is None:
            text_features = self.encode_text(tokens)
        elif labels is not None and class_tokens is not None:
            class_features = self.encode_text(class_tokens)
            text_features = class_features[labels]
        else:
            raise ValueError("labels 和 class_tokens 必须同时提供")
        # (B, B)
        logits = img_features @ text_features.T * self.logit_scale.exp() # (bzI, bzT)
        
        return logits
    
    def l2_norm(self, input, dim=-1):
        return input / torch.sqrt(torch.sum(input ** 2, dim=dim, keepdim=True) + 1e-6)
