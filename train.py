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

def main():
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
    num_epochs = config.num_epochs
    batch_size = config.batch_size

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

    model = clip(img_size, patch_size, in_chans, embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length, vocab_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 类别用于推理
    class_tokens = torch.stack([
        test_ds.encode_text(prompt)
        for prompt in test_ds.prompts
    ]).to(device)

    optimizer = create_optimizer(model)
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0.0,  # 最低学习率
    )

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, scheduler)
        
        acc = evaluate(model, test_loader, device, class_tokens)
        
    print(f"Epoch {epoch}: loss={avg_loss:.4f}, classification acc={acc:.4f}")

def train_one_epoch(model, optimizer, train_loader, device, scheduler):
    # 训练流程
    total_loss = 0.0
    total_samples = 0
    model.train()
    for imgs, tokens, labels in train_loader:
        optimizer.zero_grad()
        
        imgs = imgs.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(imgs, tokens)
        loss = contrastive_loss(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=4.605170186)  # 原论文 ln(100) 温度约束
        total_samples += imgs.size(0)
        total_loss += loss.item() * imgs.size(0)
    
        scheduler.step()  
        
    return total_loss / total_samples

def evaluate(model, test_loader, device, class_tokens):
     # 推理评估
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, _, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            logits = model(imgs, class_tokens)  # [B, 10]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    return correct / total

if __name__ == "__main__":
    main()