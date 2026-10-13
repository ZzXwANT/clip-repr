from model.clip import clip
from loss import contrastive_loss
from optim import create_optimizer

import torch
from torch.utils.data import DataLoader
from config import config
from data.dataset import Dataset

import os
path = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(path, "data")

img_size = config.img_size
patch_size = config.patch_size
in_chans = config.in_chans
vocab_size = config.vocab_size
context_length = config.context_length
embed_dims = config.embed_dims
num_heads = config.num_heads
hidden_dims = config.hidden_dims
num_layers = config.num_layers
output_dims = config.output_dims
learning_rate = config.learning_rate
num_epochs = config.num_epochs
weight_decay = config.weight_decay
batch_size = config.batch_size

model = clip(img_size, patch_size, in_chans, embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length, vocab_size)

optimizer = create_optimizer(model)

train_ds = Dataset(data_root, True, img_size, context_length, vocab_size)
test_ds = Dataset(data_root, False, img_size, context_length, vocab_size)

train_loader = DataLoader(
    train_ds,
    batch_size, 
    shuffle=True, 
    num_workers=0,
    drop_last=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size, 
    shuffle=False, 
    num_workers=0,
    drop_last=False,
)

total_steps = num_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=0.0,  # 最低学习率
)

for epoch in range(num_epochs):
    for imgs, tokens, labels in train_loader:
        optimizer.zero_grad()
        
        logits = model(imgs, tokens)
        loss = contrastive_loss(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=4.605170186)  # 原论文 ln(100) 约束
        scheduler.step()  
        
print("")
