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
from tqdm import tqdm
import logging
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import random, numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    seed = config.seed
    set_seed(seed)

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
    resume = config.resume
    num_workers = config.num_workers
    
    train_ds = Dataset(data_root, True, img_size, context_length, vocab_size)
    test_ds = Dataset(data_root, False, img_size, context_length, vocab_size)

    train_loader = DataLoader(
        train_ds,
        batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
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
    
    output_dir = Path(path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
    )

    start_epoch = 0
    best_acc = -1.0
    
    if resume:
        checkpoint = torch.load(
        output_dir / "last.pt",
        map_location=device,
        weights_only=True,
    )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        
    writer = SummaryWriter(
        log_dir=str(output_dir / "tensorboard"),
        purge_step=start_epoch + 1 if resume else None,
    )
    
    for epoch in range(start_epoch, num_epochs):
        started = time.perf_counter()

        avg_loss = train_one_epoch(model, optimizer, train_loader, device, scheduler, epoch)
        
        acc = evaluate(model, test_loader, device, class_tokens)
        logging.info(
            "Epoch %d/%d | loss=%.4f | acc=%.2f%% | lr=%.2e | time=%.1fs",
            epoch + 1, num_epochs,
            avg_loss, acc * 100,
            optimizer.param_groups[0]["lr"],
            time.perf_counter() - started,
        )
        
        step = epoch + 1

        writer.add_scalar("train/loss", avg_loss, step)
        writer.add_scalar("eval/accuracy", acc, step)
        writer.add_scalar(
            "train/lr", optimizer.param_groups[0]["lr"], step
        )
        writer.add_scalar(
            "train/logit_scale", model.logit_scale.exp().item(), step
        )
        writer.flush()
        
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }

        torch.save(checkpoint, output_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")
    writer.close()
    
def train_one_epoch(model, optimizer, train_loader, device, scheduler, epoch):
    # 训练流程
    total_loss = 0.0
    total_samples = 0
    model.train()
    for imgs, tokens, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad(set_to_none=True)
        
        # pinned memory 配合 non_blocking=True 可以减少不必要的搬运同步
        imgs = imgs.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # 计算过程bf16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(imgs, tokens)
        # loss保持fp32
        loss = contrastive_loss(logits.float(), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=4.605170186)  # 原论文 ln(100) 温度约束
        total_samples += imgs.size(0)
        total_loss += loss.item() * imgs.size(0)
    
        scheduler.step()  
        
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, test_loader, device, class_tokens):
     # 推理评估
    correct = 0
    total = 0
    model.eval()
    
    # 共用文本特征，减小计算
    class_features = model.encode_text(class_tokens)
    scale = model.logit_scale.exp()
    
    with torch.no_grad():
        for imgs, _, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            img_features = model.encode_image(imgs)
            logits = img_features @ class_features.T * scale
            # [B, 10]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    return correct / total

if __name__ == "__main__":
    main()