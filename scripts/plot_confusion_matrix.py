"""Plot CIFAR-10 test-set confusion matrices for a saved CLIP experiment.

Run from the project directory:
    python scripts/plot_confusion_matrix.py

CIFAR-10 has no separate official validation split; this uses train=False,
the same test split used by train.py for eval/accuracy.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def update_confusion_matrix(matrix, labels, predictions):
    """Accumulate all examples, including repeated (true, predicted) pairs."""
    np.add.at(matrix, (labels, predictions), 1)


def save_confusion_matrix(matrix, class_names, output_dir, title):
    import matplotlib

    matplotlib.use("Agg")  # AutoDL servers do not need a graphical display.
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    for filename, values in (
        ("confusion_matrix.csv", matrix),
        ("confusion_matrix_normalized.csv", normalized),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true / predicted", *class_names])
            for name, row in zip(class_names, values):
                writer.writerow([name, *row.tolist()])

    fig, axes = plt.subplots(1, 2, figsize=(21, 9), constrained_layout=True)
    for ax, values, panel_title, is_normalized in (
        (axes[0], matrix, "Image counts", False),
        (axes[1], normalized, "Normalized by true class (row)", True),
    ):
        vmax = 1.0 if is_normalized else max(int(matrix.max()), 1)
        heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), labels=class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(panel_title)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = values[i, j]
                label = f"{value:.1%}" if is_normalized else str(int(value))
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if value > vmax / 2 else "black")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title}\nCIFAR-10 test | images={total} | Top-1 accuracy={accuracy:.2%}")
    try:
        fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    finally:
        plt.close(fig)
    return accuracy


def collect_confusion_matrix(model, loader, device, class_tokens, image_preprocess=None):
    import torch
    from tqdm import tqdm

    model.eval()
    num_classes = class_tokens.shape[0]
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.inference_mode():
        class_features = model.encode_text(class_tokens.to(device))
        scale = model.logit_scale.exp()
        for images, _, labels in tqdm(loader, desc="CIFAR-10 confusion matrix"):
            images = images.to(device, non_blocking=True)
            if image_preprocess is not None:
                images = image_preprocess(images)
            logits = model.encode_image(images) @ class_features.T * scale
            predictions = logits.argmax(dim=1).cpu().numpy()
            update_confusion_matrix(matrix, labels.cpu().numpy(), predictions)
    return matrix


def main(args):
    # Direct script execution puts scripts/, rather than the project, on sys.path.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import Dataset
    from data.preprocess import BatchImagePreprocess
    from model.clip import clip
    from zero_shot import load_config

    checkpoint_path = args.checkpoint.expanduser().resolve()
    for path in (checkpoint_path, checkpoint_path.parent / "config.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")
    data_root = args.data_root.expanduser().resolve()
    if not (data_root / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"Expected CIFAR-10 data in {data_root / 'cifar-10-batches-py'}")
    config = load_config(checkpoint_path)
    training_dataset = getattr(config, "dataset", "cifar10")
    if training_dataset != "cifar10":
        raise ValueError(f"Expected a CIFAR-10 experiment; config.dataset={training_dataset!r}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto")

    # Dataset supplies the same class order, prompts and tokenization as training.
    # Data is checked above; a valid local CIFAR-10 dataset is used without download.
    dataset = Dataset(
        data_root, train=False, aug=False, img_size=config.img_size,
        context_length=config.context_length, vocab_size=config.vocab_size,
        dataset_name="cifar10", raw_images=args.preprocess == "tensor",
    )
    image_preprocess = (
        BatchImagePreprocess(config.img_size, "cifar10", aug=False).to(device)
        if args.preprocess == "tensor" else None
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, drop_last=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    del checkpoint
    model = clip(
        config.img_size, config.patch_size, config.in_chans,
        config.embed_dims_i, config.embed_dims_t, config.num_heads_i,
        config.num_heads_t, config.hidden_dims_i, config.hidden_dims_t,
        config.num_layers, config.output_dims, config.context_length, config.vocab_size,
        image_dropout=getattr(config, "image_dropout", 0.0),
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model = model.to(device).eval()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | preprocess: {args.preprocess} | images: {len(dataset)}")
    matrix = collect_confusion_matrix(
        model, loader, device, dataset.class_tokens, image_preprocess,
    )
    if int(matrix.sum()) != len(dataset):
        raise RuntimeError("Confusion matrix count does not match dataset size")
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else checkpoint_path.parent / "confusion_matrix"
    )
    accuracy = save_confusion_matrix(matrix, dataset.classes, output_dir, checkpoint_path.parent.name)
    print(f"Top-1 accuracy: {accuracy:.2%}")
    print("Rows = true class; columns = predicted class")
    for name, row, diagonal in zip(dataset.classes, matrix, matrix.diagonal()):
        support = int(row.sum())
        recall = float(diagonal / support) if support else 0.0
        print(f"  {name:>10}: recall={recall:.2%} ({diagonal}/{support})")
    print(f"Saved confusion_matrix.png, confusion_matrix.csv, "
          f"confusion_matrix_normalized.csv to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("~/autodl-tmp/model/cifar-100_vit-base-224-32/exp13_dropout/best.pt"),
        help="实验权重路径；同目录需要 config.py",
    )
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output_dir", type=Path, help="默认保存到权重目录下的 confusion_matrix/")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--preprocess", choices=("tensor", "pil"), default="tensor",
        help="tensor 对应 train.py --eval_preprocess cuda；pil 对应 zero_shot.py",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
