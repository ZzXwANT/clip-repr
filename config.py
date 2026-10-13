class config:
    img_size = 224
    patch_size = 32 
    in_chans = 3
    vocab_size = 49152 + 256
    context_length = 77
    embed_dims = 768
    num_heads = 12
    hidden_dims = 2048
    num_layers = 12
    output_dims = 512
    
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 40
    weight_decay = 0.01  
    resume = False