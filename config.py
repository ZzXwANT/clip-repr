class config:
    img_size = 32
    patch_size = 4 
    in_chans = 3
    vocab_size = 49152 + 256
    context_length = 16
    embed_dims = 768
    num_heads = 12
    hidden_dims = 3072
    num_layers = 12
    output_dims = 512
    
    seed = 42
    num_workers = 8
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 100
    weight_decay = 0.01  
    resume = False
    
    warmup_epochs = 5 # 添加 warmup 稳定前期训练