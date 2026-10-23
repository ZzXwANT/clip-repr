r"""Train OpenCLIP ViT-B/32 from scratch on CIFAR with OpenCLIP components.

    python scripts/train_open_clip.py --dataset cifar10 --n 18_open_clip \
        --save_dir ~/autodl-tmp/model/cifar-10_vit-base-224-32
    python scripts/train_open_clip.py --dataset cifar100 --n 19_open_clip \
        --save_dir ~/autodl-tmp/model/cifar-100_vit-base-224-32
"""

import argparse
import csv
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
import open_clip
from open_clip import ClipLoss
from open_clip_train.scheduler import cosine_lr

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ViT-B-32"
PIPELINE = "open_clip_native_v1"
METRICS = ("epoch", "loss", "train_acc", "test_acc", "lr", "logit_scale", "seconds")


def create_model_and_transforms(context_length, device):
    # Both towers and the token embedding are randomly initialized. The BPE
    # vocabulary is packaged with OpenCLIP and does not contain model weights.
    model, train_transform, eval_transform = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=None, load_weights=False,
        pretrained_image=False, pretrained_text=False,
        force_context_length=context_length, force_quick_gelu=True,
        force_patch_dropout=0.0, precision="fp32", device=device,
    )
    if model.context_length != context_length or tuple(model.positional_embedding.shape) != (context_length, 512):
        raise RuntimeError("Unexpected text context / positional embedding shape")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and module.p != 0:
            raise RuntimeError(f"Dropout must be disabled: {name}")
        if isinstance(module, nn.MultiheadAttention) and module.dropout != 0:
            raise RuntimeError(f"Attention dropout must be disabled: {name}")
    return model, train_transform, eval_transform


def create_optimizer(model, learning_rate, weight_decay, no_logit_scale_wd=True):
    # Same grouping as OpenCLIP's default AdamW branch in open_clip_train/main.py.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        excluded = (param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name)
        if name == "logit_scale":
            excluded = no_logit_scale_wd
        (no_decay if excluded else decay).append(param)
    return torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0},
         {"params": decay, "weight_decay": weight_decay}],
        lr=learning_rate, betas=(0.9, 0.98), eps=1e-6,
    )


def create_data(args, device, train_transform, eval_transform):
    # OpenCLIP has no CIFAR reader; torchvision supplies images and class labels,
    # while OpenCLIP supplies the tokenizer and all image preprocessing.
    dataset_class = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    common = dict(root=args.data_root, download=True)
    train_ds = dataset_class(train=True, transform=train_transform, **common)
    train_eval_ds = dataset_class(train=True, transform=eval_transform, **common)
    test_ds = dataset_class(train=False, transform=eval_transform, **common)
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in train_ds.classes]
    tokenizer = open_clip.get_tokenizer(MODEL_NAME, context_length=args.context_length)
    class_tokens = tokenizer(prompts)
    train_data, train_eval_data, test_data = train_ds, train_eval_ds, test_ds
    if args.smoke_test:
        train_data = Subset(train_ds, range(min(2 * args.batch_size, len(train_ds))))
        # Include an incomplete final batch to exercise accuracy accumulation.
        train_eval_data = Subset(train_eval_ds, range(min(args.batch_size + 1, len(train_eval_ds))))
        test_data = Subset(test_ds, range(min(args.batch_size + 1, len(test_ds))))
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda",
                       persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True,
                              generator=generator,
                              prefetch_factor=4 if args.num_workers > 0 else None,
                              **loader_args)
    train_eval_loader = DataLoader(train_eval_data, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **loader_args)
    if not len(train_loader):
        raise ValueError("batch_size exceeds the training set size")
    return (train_loader, train_eval_loader, test_loader, class_tokens.to(device),
            generator, train_ds.classes, prompts)


def train_one_epoch(model, loader, tokens, loss_fn, optimizer, scheduler, device, precision, epoch, global_step):
    model.train()
    total_loss = torch.zeros((), device=device)
    total = 0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        scheduler(global_step)
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            image_features = model.encode_image(images, normalize=True)
            class_features = model.encode_text(tokens, normalize=True)
            # Repeated classes share exactly the same text features. With no
            # dropout, native ClipLoss has the same batch-mean objective and
            # model gradients as the former uniform multi-positive loss.
            loss = loss_fn(image_features, class_features[labels], model.logit_scale.exp())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(100))
        global_step += 1
        total_loss += loss.detach() * images.shape[0]
        total += images.shape[0]
    optimizer.zero_grad(set_to_none=True)
    return total_loss.item() / total, global_step


@torch.no_grad()
def evaluate(model, loader, tokens, device):
    # Evaluate in FP32 using OpenCLIP's deterministic validation transforms.
    model.eval()
    class_features = model.encode_text(tokens, normalize=True)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.logit_scale.exp() * model.encode_image(images, normalize=True) @ class_features.T
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        correct += (logits.argmax(dim=1) == labels).sum()
        total += labels.numel()
    return correct.item() / total


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu --precision fp32")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use --precision fp32")
    if args.resume:
        if not (args.save_dir / "last.pt").is_file():
            raise FileNotFoundError(args.save_dir / "last.pt")
    elif args.save_dir.exists() and any(
        path.name != "smoke" or not path.is_dir() for path in args.save_dir.iterdir()
    ):
        raise FileExistsError(f"Output directory is not empty: {args.save_dir}; use --resume or a new --save_dir")
    if not args.resume and args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(f"TensorBoard directory is not empty: {args.tensorboard_dir}; use --resume or a new --n")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.save_dir / f"train{args.n}.log", encoding="utf-8")])

    model, train_transform, eval_transform = create_model_and_transforms(args.context_length, device)
    (train_loader, train_eval_loader, test_loader, tokens,
     generator, classes, prompts) = create_data(args, device, train_transform, eval_transform)
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay, args.no_logit_scale_wd)
    loss_fn = ClipLoss(cache_labels=True)
    settings = {name: getattr(args, name) for name in (
        "dataset", "context_length", "batch_size", "learning_rate", "weight_decay",
        "no_logit_scale_wd", "num_epochs", "warmup_epochs", "precision", "seed",
        "smoke_test",
    )}
    settings.update(open_clip_version=importlib.metadata.version("open_clip_torch"),
                    steps_per_epoch=len(train_loader), pipeline=PIPELINE)
    model_config = open_clip.get_model_config(MODEL_NAME)
    model_config["text_cfg"]["context_length"] = args.context_length
    model_config["vision_cfg"]["patch_dropout"] = 0.0
    model_config["quick_gelu"] = True
    metadata = dict(settings=settings, model_name=MODEL_NAME, model_config=model_config,
                    pretrained=None, dropout=0.0, data_root=str(args.data_root),
                    classes=classes, prompts=prompts, class_tokens=tokens.cpu().tolist(),
                    preprocess_cfg=open_clip.get_model_preprocess_cfg(model),
                    train_transform=str(train_transform), eval_transform=str(eval_transform),
                    loss="open_clip.ClipLoss", scheduler="open_clip_train.scheduler.cosine_lr",
                    optimizer={"name": "AdamW", "betas": [0.9, 0.98], "eps": 1e-6},
                    parameters=sum(param.numel() for param in model.parameters()),
                    train_samples=len(train_loader.dataset),
                    train_eval_samples=len(train_eval_loader.dataset),
                    test_samples=len(test_loader.dataset), device=str(device),
                    torch_version=str(torch.__version__))
    scheduler_config = dict(name="open_clip_train.scheduler.cosine_lr", base_lr=args.learning_rate,
                            warmup_steps=args.warmup_epochs * len(train_loader),
                            total_steps=args.num_epochs * len(train_loader))
    scheduler = cosine_lr(optimizer, args.learning_rate, scheduler_config["warmup_steps"],
                          scheduler_config["total_steps"])
    start_epoch, best_acc, global_step = 0, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.save_dir / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("metadata", {}).get("settings", {}).get("pipeline") != PIPELINE:
            raise ValueError("This checkpoint used the former custom pipeline. Start a new --n for the OpenCLIP native pipeline.")
        if checkpoint["metadata"]["settings"] != settings:
            raise ValueError("Resume settings differ from config.json; use the original dataset, hyperparameters and total epochs")
        if checkpoint["scheduler"] != scheduler_config:
            raise ValueError("Resume learning-rate schedule differs from the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], generator)
        start_epoch, best_acc = checkpoint["epoch"] + 1, checkpoint["best_acc"]
        global_step = checkpoint["global_step"]
        if global_step != start_epoch * len(train_loader):
            raise ValueError("Checkpoint global_step does not match its completed epochs")
        del checkpoint
        # Discard CSV rows written after the latest completed checkpoint.
        csv_path = args.save_dir / "metrics.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) <= start_epoch]
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=METRICS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    else:
        (args.save_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("%s | random initialization | QuickGELU | dropout=0 | context=%d | parameters=%d",
                 MODEL_NAME, args.context_length, metadata["parameters"])
    logging.info("%s | class tokens=%s | batch=%d | logit_scale WD=%s | output=%s",
                 args.dataset, tuple(tokens.shape), args.batch_size, not args.no_logit_scale_wd, args.save_dir)
    logging.info("TensorBoard: %s", args.tensorboard_dir)
    logging.info("OpenCLIP native transforms (CPU workers), ClipLoss, cosine_lr | global_step=%d", global_step)
    if args.smoke_test:
        logging.info("SMOKE TEST ONLY: two epochs on subsets; these are not experiment results")
    if start_epoch >= args.num_epochs:
        logging.info("Target epochs already completed (%d)", start_epoch)
        return
    end_epoch = args.num_epochs
    if args.max_epochs_this_run is not None:
        end_epoch = min(end_epoch, start_epoch + args.max_epochs_this_run)
    with SummaryWriter(str(args.tensorboard_dir), purge_step=start_epoch + 1 if args.resume else None) as writer:
        for epoch in range(start_epoch, end_epoch):
            started = time.perf_counter()
            loss, global_step = train_one_epoch(model, train_loader, tokens, loss_fn, optimizer,
                                                scheduler, device, args.precision, epoch + 1, global_step)
            train_acc = evaluate(model, train_eval_loader, tokens, device)
            test_acc = evaluate(model, test_loader, tokens, device)
            lr, scale = optimizer.param_groups[0]["lr"], model.logit_scale.exp().item()
            elapsed = time.perf_counter() - started
            row = dict(zip(METRICS, (epoch + 1, loss, train_acc, test_acc, lr, scale, elapsed)))
            logging.info("Epoch %d/%d | loss=%.4f | train_acc=%.2f%% | test_acc=%.2f%% | lr=%.2e | logit_scale=%.4f | time=%.1fs",
                         epoch + 1, args.num_epochs, loss, 100 * train_acc, 100 * test_acc, lr, scale, elapsed)
            csv_path = args.save_dir / "metrics.csv"
            new_csv = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=METRICS, lineterminator="\n")
                if new_csv:
                    csv_writer.writeheader()
                csv_writer.writerow(row)
            for tag, value in (("train/loss", loss), ("train/accuracy", train_acc),
                               ("eval/accuracy", test_acc), ("train/lr", lr), ("train/logit_scale", scale)):
                writer.add_scalar(tag, value, epoch + 1)
            writer.flush()
            is_best = test_acc > best_acc
            best_acc = max(best_acc, test_acc)
            state = dict(epoch=epoch, model=model.state_dict(), best_acc=best_acc, metadata=metadata)
            if is_best:
                save_checkpoint(state, args.save_dir / "best.pt")
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler_config,
                         global_step=global_step, rng=rng_state(generator))
            save_checkpoint(state, args.save_dir / "last.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data")
    parser.add_argument("--n", type=str, required=True, help="Experiment number / suffix, as in train.py; outputs use exp<n>")
    parser.add_argument("--save_dir", type=Path, required=True, help="Model parent directory, as in train.py; checkpoints go to save_dir/exp<n>")
    parser.add_argument("--context_length", type=int, help="Defaults: CIFAR-10=8, CIFAR-100=16, including SOT/EOT")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    scale_wd = parser.add_mutually_exclusive_group()
    scale_wd.add_argument("--no_logit_scale_wd", dest="no_logit_scale_wd", action="store_true", default=True,
                          help="Exclude logit_scale from WD (OpenCLIP default)")
    scale_wd.add_argument("--logit_scale_wd", dest="no_logit_scale_wd", action="store_false",
                          help="Apply weight decay to logit_scale instead of OpenCLIP's default")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume save_dir/exp<n>/last.pt using the original total epochs and settings")
    parser.add_argument("--max_epochs_this_run", type=int, help="Stop cleanly after N epochs in this invocation without changing the LR schedule")
    parser.add_argument("--smoke_test", action="store_true", help="Two epochs, one warmup epoch, at most two training batches/epoch and batch_size+1 eval images; models in save_dir/exp<n>/smoke, TensorBoard in outputs/tensorboard/smoke/exp<n>")
    args = parser.parse_args(argv)
    if not args.n or "/" in args.n or "\\" in args.n:
        parser.error("n must be a nonempty experiment number / suffix without path separators")
    if args.context_length is None:
        args.context_length = 8 if args.dataset == "cifar10" else 16
    if args.smoke_test:
        args.num_epochs, args.warmup_epochs = 2, 1
    if args.context_length < 2 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("context_length and batch_size must be >= 2; num_workers must be >= 0")
    if not 0 <= args.warmup_epochs < args.num_epochs:
        parser.error("Require 0 <= warmup_epochs < num_epochs")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning_rate must be positive; weight_decay must be nonnegative")
    if args.max_epochs_this_run is not None and args.max_epochs_this_run < 1:
        parser.error("max_epochs_this_run must be positive")
    args.data_root = args.data_root.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve() / f"exp{args.n}"
    args.tensorboard_dir = ROOT / "outputs" / "tensorboard" / f"exp{args.n}"
    if args.smoke_test:
        args.save_dir /= "smoke"
        args.tensorboard_dir = ROOT / "outputs" / "tensorboard" / "smoke" / f"exp{args.n}"
    return args


if __name__ == "__main__":
    main(parse_args())
