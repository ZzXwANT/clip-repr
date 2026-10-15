from model.clip import clip
from loss import contrastive_loss
from optim import create_optimizer
from scheduler import create_scheduler, restore_scheduler

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
import argparse

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):
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
    
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    train_ds = Dataset(data_root, True, img_size, context_length, vocab_size)
    test_ds = Dataset(data_root, False, img_size, context_length, vocab_size)

    train_loader = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    # 模型
    model = clip(img_size, patch_size, in_chans, embed_dims, num_heads, hidden_dims, num_layers, output_dims, context_length, vocab_size)
    
    model = model.to(device)

    # 类别用于推理
    class_tokens = torch.stack([
        test_ds.encode_text(prompt)
        for prompt in test_ds.prompts
    ]).to(device)
    
    # record
    output_dir = Path(path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / f"train{args.n}.log", encoding="utf-8"),
        ],
    )
    logging.info("Training precision: %s", args.precision)

    start_epoch = 0
    best_acc = -1.0

    # optimizer
    optimizer = create_optimizer(model, args.learning_rate)

    if args.resume:
        checkpoint = torch.load(
            args.model_dir / "last.pt",
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model"])

        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]

    if start_epoch >= args.num_epochs:
        logging.info("已达到目标训练轮数，无需继续训练")
        return

    if args.resume and not args.reset_scheduler:
        # 先创建调度器，再恢复优化器，保留 checkpoint 中的当前学习率
        scheduler = restore_scheduler(optimizer, checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
                group["initial_lr"] = args.learning_rate
        steps_per_epoch = len(train_loader)
        total_steps = (args.num_epochs - start_epoch) * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        scheduler = create_scheduler(optimizer, total_steps, warmup_steps)

    writer = SummaryWriter(
        log_dir=str(output_dir / "tensorboard" / f"exp{args.n}"),
        purge_step=start_epoch + 1 if args.resume else None,
    )
    
    # 训练 loop
    for epoch in range(start_epoch, args.num_epochs):
        started = time.perf_counter()

        avg_loss = train_one_epoch(model, optimizer, train_loader, device, scheduler, epoch, args.precision)
        
        acc = evaluate(model, test_loader, device, class_tokens)
        
        logging.info(
            "Epoch %d/%d | loss=%.4f | acc=%.2f%% | lr=%.2e | time=%.1fs",
            epoch + 1, args.num_epochs,
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

        torch.save(checkpoint, args.save_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, args.save_dir / "best.pt")
    writer.close()
    
def train_one_epoch(model, optimizer, train_loader, device, scheduler, epoch, precision):
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
        # 根据 --precision 控制混合精度
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
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
            preds = logits.argmax(dim=1)        # 贪心选择
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="实验次数/备注")
    parser.add_argument("--save_dir", type=Path, required=True, help="模型保存路径")
    parser.add_argument("--resume", action="store_true", help="恢复模型、优化器和原学习率计划")
    parser.add_argument("--reset_scheduler", action="store_true", help="续训时按本次学习率和 warmup 重建调度器")
    parser.add_argument("--model_dir", type=Path, help="续训模型目录，包含 last.pt；--resume 时必填")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16", help="训练精度，默认 bf16；设备不支持时使用 fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()
    if args.resume and not (args.model_dir and (args.model_dir / "last.pt").is_file()): parser.error("续训时 --model_dir 必须指向包含 last.pt 的目录")
    if (not args.resume or args.reset_scheduler) and not 0 <= args.warmup_epochs < args.num_epochs: parser.error("需要满足 0 <= warmup_epochs < num_epochs")
    if args.batch_size <= 0 or args.learning_rate <= 0 or args.num_epochs <= 0: parser.error("batch_size、learning_rate 和 num_epochs 必须大于 0")
    if args.num_workers < 0: parser.error("num_workers 不能小于 0")
    if args.precision == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()): args.precision = "fp32"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "info.txt").write_text(
        f"seed: {args.seed}\n"
        f"batch_size: {args.batch_size}\n"
        f"num_workers: {args.num_workers}\n"
        f"learning_rate: {args.learning_rate}\n"
        f"num_epochs: {args.num_epochs}\n"
        f"warmup_epochs: {args.warmup_epochs}\n"
        f"precision: {args.precision}\n"
        f"reset_scheduler: {args.reset_scheduler}\n",
        encoding="utf-8",
    )
    main(args)
