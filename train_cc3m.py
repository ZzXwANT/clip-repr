"""Single-device CLIP pretraining on local/Drive CC3M tar shards."""
import argparse
from contextlib import nullcontext
import json
import logging
import math
from pathlib import Path
import random
import runpy
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from data.cc3m import CC3MDataset, CLIP_MEAN, CLIP_STD, shard_manifest, split_shards
from loss import paired_contrastive_loss
from model.clip import clip
from optim import create_optimizer
from scheduler import create_scheduler, restore_scheduler

MODEL_FIELDS = (
    "img_size", "patch_size", "in_chans", "embed_dims_i", "embed_dims_t",
    "num_heads_i", "num_heads_t", "hidden_dims_i", "hidden_dims_t",
    "num_layers", "output_dims", "context_length", "vocab_size",
)


def atomic_save(checkpoint, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(target)


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() == source.resolve():
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def sync_file(source, directory):
    if directory is not None:
        atomic_copy(source, Path(directory) / source.name)


def autocast(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, precision, max_samples):
    """Both directions use the entire fixed validation pool as candidates."""
    model.eval()
    images, texts = [], []
    count = 0
    for batch_images, batch_tokens in loader:
        remaining = max_samples - count
        batch_images, batch_tokens = batch_images[:remaining], batch_tokens[:remaining]
        with autocast(device, precision):
            images.append(model.encode_image(batch_images.to(device, non_blocking=True)).float())
            texts.append(model.encode_text(batch_tokens.to(device, non_blocking=True)).float())
        count += len(batch_images)
        if count >= max_samples:
            break
    if count < 2:
        raise ValueError("验证集至少需要两个有效图文对")
    images, texts = torch.cat(images), torch.cat(texts)
    scale = model.logit_scale.float().exp()
    metrics = {"samples": count}
    loss_sum = 0.0
    for name, queries, candidates in (("i2t", images, texts), ("t2i", texts, images)):
        correct = {k: 0 for k in (1, 5, 10)}
        for start in range(0, count, 256):
            logits = queries[start:start + 256] @ candidates.T * scale
            target = torch.arange(start, start + len(logits), device=device)
            loss_sum += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
            top = logits.topk(min(10, count), dim=1).indices
            for k in correct:
                correct[k] += (top[:, :min(k, count)] == target[:, None]).any(dim=1).sum().item()
        metrics.update({f"{name}_R@{k}": value / count for k, value in correct.items()})
    metrics["loss"] = loss_sum / (2 * count)
    model.train()
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, help="可选，必须与训练目录互不包含")
    parser.add_argument("--val-shards", type=int, default=1, help="未指定 val-root 时固定留出的分片数")
    parser.add_argument("--save-dir", type=Path, required=True, help="Colab 本地实验目录")
    parser.add_argument("--drive-save-dir", type=Path, help="同步 checkpoint 和日志到 Drive 的实验目录")
    parser.add_argument("--cache-dir", type=Path, help="每个加载进程缓存当前 tar 的本地目录")
    parser.add_argument("--resume", type=Path, help="恢复本入口生成的 last.pt")
    parser.add_argument("--model-config", type=Path, help="可选 Python 配置文件，定义 config 类")
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=30000, help="完整学习率计划的优化器更新次数")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--stop-after-steps", type=int, help="本次会话最多更新多少步，不改变总学习率计划")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(args):
    if (args.batch_size < 2 or args.num_workers < 0 or args.context_length < 2
            or args.shuffle_buffer < 0 or args.val_samples < 2):
        raise ValueError("batch-size、context-length、val-samples >= 2；workers 和 buffer >= 0")
    if not 0 <= args.warmup_steps < args.total_steps:
        raise ValueError("需要 0 <= warmup-steps < total-steps")
    if min(args.save_every, args.eval_every, args.log_every) < 1:
        raise ValueError("保存、评估和日志间隔必须为正数")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop-after-steps 必须为正数")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate > 0，weight-decay >= 0")
    args.save_dir = args.save_dir.expanduser().resolve()
    if (args.save_dir / "last.pt").exists() and not args.resume:
        raise ValueError("实验目录已有 last.pt；请指定 --resume 或使用新的 save-dir")
    if args.drive_save_dir:
        args.drive_save_dir = args.drive_save_dir.expanduser().resolve()
        if (args.drive_save_dir / "last.pt").exists() and not args.resume:
            raise ValueError("Drive 实验目录已有 last.pt；请恢复或使用新的实验目录")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.drive_save_dir and not (args.save_dir / "train.log").exists():
        previous_log = args.drive_save_dir / "train.log"
        if previous_log.exists():
            sync_file(previous_log, args.save_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(args.save_dir / "train.log", encoding="utf-8"),
    ], force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("没有可用 CUDA GPU")
    if args.precision == "auto":
        args.precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 模式需要支持 BF16 的 CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    source_config = runpy.run_path(str(args.model_config))["config"] if args.model_config else config
    model_config = {key: getattr(source_config, key) for key in MODEL_FIELDS}
    model_config["context_length"] = args.context_length
    train_shards, val_shards = split_shards(args.data_root, args.val_root, args.val_shards, args.seed)
    manifest = {
        "train": shard_manifest(train_shards, args.data_root),
        "val": shard_manifest(val_shards, args.val_root or args.data_root),
    }
    # Keep validation single-process: a fixed prefix gives the same pool regardless
    # of the number of training workers and across resumed sessions.
    workers = min(args.num_workers, len(train_shards))
    dataset_kwargs = {k: model_config[k] for k in ("img_size", "context_length", "vocab_size")}
    train_ds = CC3MDataset(train_shards, **dataset_kwargs, seed=args.seed,
                           shuffle_buffer=args.shuffle_buffer, cache_dir=args.cache_dir)
    val_ds = CC3MDataset(val_shards, **dataset_kwargs, train=False, cache_dir=args.cache_dir)
    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, **loader_kwargs, num_workers=workers,
                             drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, **loader_kwargs, num_workers=0)
    model = clip(**model_config).to(device)
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    global_step, data_epoch, best_loss = 0, 0, math.inf
    checkpoint = None
    resume_settings = {k: getattr(args, k) for k in (
        "batch_size", "total_steps", "warmup_steps", "seed", "precision", "shuffle_buffer",
        "learning_rate", "weight_decay", "val_samples",
    )}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != model_config or checkpoint["manifest"] != manifest:
            raise ValueError("恢复失败：模型配置或数据分片清单发生变化")
        if checkpoint["resume_settings"] != resume_settings:
            raise ValueError("恢复时训练参数必须与原训练一致；检查 batch、步数计划、精度、学习率和验证池大小")
        model.load_state_dict(checkpoint["model"])
        scheduler = restore_scheduler(optimizer, checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step, best_loss = checkpoint["global_step"], checkpoint["best_loss"]
        data_epoch = checkpoint["data_epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        logging.info("恢复 step=%d；数据从下一轮分片打乱开始，未恢复 worker/样本游标", global_step)
        del checkpoint
    else:
        scheduler = create_scheduler(optimizer, args.total_steps, args.warmup_steps)
    run_config = {
        "model": model_config, "image_mean": CLIP_MEAN, "image_std": CLIP_STD,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resume_note": "Optimizer/LR step restored; restart shuffled data pass, not exact sample cursor.",
    }
    (args.save_dir / "run.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.save_dir / "shards.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    saved_config = "class config:\n" + "".join(f"    {k} = {v!r}\n" for k, v in model_config.items())
    saved_config += f"    image_mean = {CLIP_MEAN!r}\n    image_std = {CLIP_STD!r}\n    clip_preprocess = True\n"
    (args.save_dir / "config.py").write_text(saved_config, encoding="utf-8")
    for filename in ("run.json", "shards.json", "config.py"):
        sync_file(args.save_dir / filename, args.drive_save_dir)
    logging.info("Device=%s | precision=%s | train shards=%d | val shards=%d | workers=%d",
                 device, args.precision, len(train_shards), len(val_shards), workers)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logging.info("GPU=%s | VRAM=%.1f GiB", props.name, props.total_memory / 2**30)
    logging.info("%s；physical batch=%d，每步负样本来自此 batch",
                 "恢复已有 CC3M checkpoint" if args.resume else "从头训练", args.batch_size)
    target_step = min(args.total_steps, global_step + args.stop_after_steps) if args.stop_after_steps else args.total_steps
    if global_step >= target_step:
        logging.info("已达到目标步数")
        return
    writer = SummaryWriter(str(args.save_dir / "tensorboard"), purge_step=global_step + 1 if args.resume else None)

    def save(is_best=False):
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "global_step": global_step,
            "data_epoch": data_epoch, "best_loss": best_loss,
            "model_config": model_config, "manifest": manifest, "resume_settings": resume_settings,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        atomic_save(state, args.save_dir / "last.pt")
        if is_best:
            # Avoid serializing the full optimizer twice.
            atomic_copy(args.save_dir / "last.pt", args.save_dir / "best.pt")
        writer.flush()
        for filename in ("last.pt", "train.log"):
            sync_file(args.save_dir / filename, args.drive_save_dir)
        if is_best:
            sync_file(args.save_dir / "best.pt", args.drive_save_dir)
        if args.drive_save_dir:
            for event_file in (args.save_dir / "tensorboard").glob("events.*"):
                sync_file(event_file, args.drive_save_dir / "tensorboard")

    started = time.perf_counter()
    window_loss = window_samples = window_steps = 0
    last_log_time = started
    progress = tqdm(total=target_step, initial=global_step, desc="CC3M steps")
    model.train()
    try:
        while global_step < target_step:
            train_ds.set_epoch(data_epoch)
            batches = 0
            for images, tokens in train_loader:
                images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, args.precision):
                    logits = model(images, tokens)
                loss = paired_contrastive_loss(logits)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {global_step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    model.logit_scale.clamp_(max=math.log(100))
                global_step += 1
                batches += 1
                loss_value = loss.item()
                window_loss += loss_value
                window_samples += len(images)
                window_steps += 1
                progress.update(1)
                if global_step % args.log_every == 0 or global_step == target_step:
                    now = time.perf_counter()
                    throughput = window_samples / (now - last_log_time)
                    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                    logging.info("step=%d/%d loss=%.4f lr=%.3g samples/s=%.1f peak_VRAM=%.2f GiB",
                                 global_step, args.total_steps, window_loss / window_steps, used_lr, throughput, peak)
                    for name, value in {"loss": window_loss / window_steps, "lr": used_lr,
                                        "samples_per_second": throughput, "peak_vram_gib": peak,
                                        "grad_norm": grad_norm.item(), "logit_scale": model.logit_scale.exp().item()}.items():
                        writer.add_scalar(f"train/{name}", value, global_step)
                    window_loss = window_samples = window_steps = 0
                    last_log_time = now
                is_best = False
                should_eval = global_step % args.eval_every == 0 or global_step == target_step
                if should_eval:
                    optimizer.zero_grad(set_to_none=True)
                    metrics = evaluate(model, val_loader, device, args.precision, args.val_samples)
                    for name, value in metrics.items():
                        writer.add_scalar(f"val/{name}", value, global_step)
                    logging.info("Validation step=%d %s", global_step, json.dumps(metrics))
                    is_best = metrics["loss"] < best_loss
                    best_loss = min(best_loss, metrics["loss"])
                if global_step % args.save_every == 0 or should_eval:
                    save(is_best)
                if global_step >= target_step:
                    break
            if batches == 0:
                raise ValueError("没有完整训练 batch；检查分片格式、有效样本数，或减小 batch-size/num-workers")
            data_epoch += 1
    except KeyboardInterrupt:
        # A CUDA interruption can leave an optimizer update half-completed. Keep
        # the last completed, periodic checkpoint rather than writing that state.
        logging.warning("训练被中断，请从最近完整写入的 last.pt 恢复")
    finally:
        progress.close()
        writer.close()
    logging.info("Finished step=%d | elapsed=%.1fs", global_step, time.perf_counter() - started)


if __name__ == "__main__":
    main(build_parser().parse_args())
