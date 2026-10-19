# 这里保存了模型的参数

class config:
    img_size = 224
    patch_size = 32 
    in_chans = 3
    vocab_size = 49152 + 256
    context_length = 8
    embed_dims_i = 384
    embed_dims_t = 256
    num_heads_i = 6
    num_heads_t = 4
    hidden_dims_i = 1536
    hidden_dims_t = 1024
    num_layers = 6
    output_dims = 256
    image_dropout = 0.1  # 图像 Transformer dropout；设为 0.0 可做基线对照
