# 这里保存了模型的参数

class config:
    img_size = 224
    patch_size = 32 
    in_chans = 3
    vocab_size = 49152 + 256
    context_length = 8
    embed_dims_i = 768
    embed_dims_t = 512
    num_heads_i = 12
    num_heads_t = 8
    hidden_dims_i = 3072
    hidden_dims_t = 2048
    num_layers = 12
    output_dims = 512
    image_dropout = 0.1  # 图像 Transformer dropout；设为 0.0 可做基线对照
