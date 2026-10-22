"""Evaluate official OpenAI CLIP ViT-B/32 on local CIFAR test sets, without training.

    python scripts/evaluate_openai_clip.py --device cuda
    python scripts/evaluate_openai_clip.py --device cpu --smoke-test

Install requirements.txt first. No data or weights are downloaded by
this script. Smoke results are isolated under output_dir/smoke/.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
MODEL_URL = f"https://openaipublic.azureedge.net/clip/models/{MODEL_SHA256}/ViT-B-32.pt"
TEMPLATE = "a photo of a {}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_outputs(matrix, classes, predictions, output_dir, dataset_name, smoke):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float),
                           where=support != 0)
    for filename, values in (("confusion_matrix.csv", matrix),
                             ("confusion_matrix_normalized.csv", normalized)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true / predicted", *classes])
            writer.writerows([name, *row.tolist()] for name, row in zip(classes, values))
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "true_label", "predicted_label"])
        writer.writerows((index, label, prediction)
                         for index, (label, prediction) in enumerate(predictions))

    large = len(classes) > 10
    fig, axes = plt.subplots(1, 2, figsize=(28, 13) if large else (21, 9),
                             constrained_layout=True)
    for ax, values, title, vmax in (
        (axes[0], matrix, "Image counts", max(1, int(matrix.max()))),
        (axes[1], normalized, "Normalized by true class (row)", 1.0),
    ):
        heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(classes)), labels=classes,
                      rotation=90 if large else 45, ha="right", fontsize=5 if large else 9)
        ax.set_yticks(range(len(classes)), labels=classes, fontsize=5 if large else 9)
        ax.set(xlabel="Predicted class", ylabel="True class", title=title)
        if not large:
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    label = f"{value:.1%}" if values is normalized else str(int(value))
                    ax.text(j, i, label, ha="center", va="center", fontsize=8,
                            color="white" if value > vmax / 2 else "black")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    accuracy = float(np.trace(matrix) / matrix.sum())
    status = "SMOKE ONLY, first 8 images" if smoke else "full test set"
    fig.suptitle(f"OpenAI CLIP ViT-B/32 | {dataset_name} | {status}\n"
                 f"images={int(matrix.sum())} | Top-1={accuracy:.2%}")
    try:
        fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    finally:
        plt.close(fig)
    return accuracy


def evaluate(model, dataset, device, args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    import clip

    # Keep torchvision's class order; turn CIFAR-100 underscores into spaces.
    prompts = [TEMPLATE.format(name.replace("_", " ")) for name in dataset.classes]
    tokens = clip.tokenize(prompts, context_length=77, truncate=False).to(device)
    samples = Subset(dataset, range(8)) if args.smoke_test else dataset
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False,
                        pin_memory=device.type == "cuda")
    matrix = np.zeros((len(prompts), len(prompts)), dtype=np.int64)
    predictions = []
    correct = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        for images, labels in tqdm(loader, desc=type(dataset).__name__):
            image_features = F.normalize(
                model.encode_image(images.to(device, non_blocking=True)).float(), dim=-1,
            )
            logits = model.logit_scale.exp().float() * image_features @ text_features.T
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite similarity scores")
            predicted = logits.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            np.add.at(matrix, (labels, predicted), 1)
            correct += int((labels == predicted).sum())
            predictions.extend(zip(labels.tolist(), predicted.tolist()))
    elapsed = time.perf_counter() - started
    expected_support = np.bincount(np.asarray(dataset.targets)[:len(samples)],
                                   minlength=len(prompts))
    if (int(matrix.sum()) != len(samples) or int(np.trace(matrix)) != correct
            or not np.array_equal(matrix.sum(axis=1), expected_support)):
        raise RuntimeError("Confusion counts do not match test labels / accuracy")
    return matrix, prompts, predictions, elapsed


def main(args):
    import numpy as np
    import torch
    import torchvision
    from torchvision.datasets import CIFAR10, CIFAR100
    import clip

    if not all(hasattr(clip, name) for name in ("load", "tokenize", "available_models")):
        raise RuntimeError("Install OpenAI's CLIP using requirements.txt")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download official weights to {checkpoint}: {MODEL_URL}")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match official OpenAI ViT-B/32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu --smoke-test")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    model.eval().requires_grad_(False)
    print(f"Checkpoint: {checkpoint}\nDevice: {device}\nPreprocess: {preprocess}", flush=True)
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    distribution = importlib.metadata.distribution("clip")
    direct_url = distribution.read_text("direct_url.json")
    environment = {name: importlib.metadata.version(name)
                   for name in ("torch", "torchvision", "numpy", "Pillow", "ftfy", "regex", "matplotlib", "clip")}
    metadata = {
        "model": "OpenAI CLIP ViT-B/32", "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODEL_SHA256, "checkpoint_url": MODEL_URL,
        "clip_source": json.loads(direct_url) if direct_url else None,
        "code_commit": code_commit, "script_sha256": sha256(Path(__file__)),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": environment, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda_version": torch.version.cuda,
        "encoder_dtype": str(model.dtype), "similarity_dtype": "torch.float32",
        "tf32": False, "prompt_template": TEMPLATE,
        "class_name_policy": "torchvision class order; replace underscores with spaces",
        "context_length": 77, "preprocessing": repr(preprocess),
        "fine_tuning": False, "prompt_ensemble": False,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "cpu_threads": args.cpu_threads,
        "confusion_axes": {"rows": "true", "columns": "predicted"},
    }
    output_root = args.output_dir.expanduser().resolve()
    if args.smoke_test:
        output_root = output_root / "smoke"
    names = ("cifar10", "cifar100") if args.dataset == "both" else (args.dataset,)
    for name in names:
        dataset_type, files = (
            (CIFAR10, ("cifar-10-batches-py/test_batch", "cifar-10-batches-py/batches.meta"))
            if name == "cifar10" else
            (CIFAR100, ("cifar-100-python/test", "cifar-100-python/meta"))
        )
        data_root = args.data_root.expanduser().resolve()
        dataset = dataset_type(str(data_root), train=False, download=False, transform=preprocess)
        matrix, prompts, predictions, elapsed = evaluate(model, dataset, device, args)
        output_dir = output_root / name
        accuracy = save_outputs(matrix, dataset.classes, predictions, output_dir, name, args.smoke_test)
        result = {
            **metadata, "dataset": name, "split": "test", "data_root": str(data_root),
            "data_sha256": {file: sha256(data_root / file) for file in files},
            "status": "smoke_only" if args.smoke_test else "full_test",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "classes": dataset.classes, "prompts": prompts,
            "total": int(matrix.sum()), "correct": int(np.trace(matrix)),
            "top1_accuracy": accuracy, "top1_pct": 100 * accuracy,
            "inference_seconds": elapsed,
            "artifacts": {path.name: sha256(path) for path in sorted(output_dir.iterdir())
                          if path.name in {"confusion_matrix.csv", "confusion_matrix_normalized.csv",
                                           "confusion_matrix.png", "predictions.csv"}},
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(f"{name}: Top-1={accuracy:.2%} ({result['correct']}/{result['total']})"
              f" | {result['status']} | {elapsed:.1f}s | {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("~/autodl-tmp/model/ViT-B-32.pt"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/results/openai_clip_vit_b32")
    parser.add_argument("--dataset", choices=("both", "cifar10", "cifar100"), default="both")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true", help="Only the first 8 test images per dataset")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.num_workers < 0:
        parser.error("batch-size/cpu-threads must be positive; num-workers must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
