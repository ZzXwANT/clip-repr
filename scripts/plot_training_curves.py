"""Redraw README figures from the checked-in training-history snapshot."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_history(path):
    experiments = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Preserve the last observation of a repeated epoch, not the maximum.
            experiments.setdefault(row["experiment"], {})[int(row["epoch"])] = row
    # Supplement the console-log snapshot with recovered full-precision events.
    for filename, experiment in (
        ("exp12_dropout.csv", "cifar-10_vit-base-224-32/exp12_dropout"),
        ("exp13_dropout_cifar-100.csv", "cifar-100_vit-base-224-32/exp13_dropout"),
    ):
        exported = path.parent / "tensorboard" / filename
        if exported.exists():
            with exported.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    experiments.setdefault(experiment, {})[int(row["epoch"])] = {
                        "train_acc_pct": str(100 * float(row["train/accuracy"])),
                        "test_acc_pct": str(100 * float(row["eval/accuracy"])),
                    }
    return experiments


def draw_curves(history, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    def plot(ax, experiment, label, metric="test_acc_pct", **kwargs):
        rows = sorted(history[experiment].items())
        values = [(epoch, float(row[metric])) for epoch, row in rows if row[metric] != ""]
        epochs, accuracy = zip(*values)
        ax.plot(epochs, accuracy, label=label, linewidth=1.7, **kwargs)

    cifar10 = "cifar-10_vit-base-224-32/"
    for experiment, label, color in (
        ("exp8-orgin-3e-4-bf16", "exp8: initial 224/32", "#94a3b8"),
        ("exp9_augI", "exp9: augmentation", "#a78bfa"),
        ("exp10_accl_weight_decay", "exp10: acceleration / WD", "#e69f00"),
        ("exp11_more_aug", "exp11: stronger augmentation", "#0072b2"),
        ("exp12_dropout", "exp12: image dropout 0.1", "#009e73"),
    ):
        plot(axes[0, 0], cifar10 + experiment, label, color=color)
    axes[0, 0].set_title("CIFAR-10: test accuracy")

    for experiment, label, color in (
        ("exp11_more_aug", "exp11", "#0072b2"),
        ("exp12_dropout", "exp12", "#009e73"),
    ):
        plot(axes[0, 1], cifar10 + experiment, label + " train", "train_acc_pct", color=color, linestyle="--")
        plot(axes[0, 1], cifar10 + experiment, label + " test", color=color)
    axes[0, 1].set_title("CIFAR-10: train / test gap")

    for experiment, label, color in (
        ("exp2_small_lr1e-4", "Small exp2: lr=1e-4", "#e69f00"),
        ("exp3_small_lr3e-4", "Small exp3: lr=3e-4", "#0072b2"),
        ("exp4_small_lr1e-3", "Small exp4: lr=1e-3", "#009e73"),
    ):
        plot(axes[1, 0], "cifar-100_vit-small-224-32/" + experiment, label, color=color)
    axes[1, 0].set_title("CIFAR-100: Small test accuracy")

    for experiment, label, color in (
        ("exp13_dropout", "Base exp13 (100 epochs, TensorBoard)", "#7b4ab5"),
        ("exp15_lr1e-4", "Base exp15 (72 epochs)", "#e69f00"),
        ("exp16_lr1e-4_adamw", "Base exp16 (46 epochs)", "#0072b2"),
        ("exp17_lr1e-4_adamw_logit_scale", "Base exp17 (76 epochs)", "#009e73"),
    ):
        plot(axes[1, 1], "cifar-100_vit-base-224-32/" + experiment, label, color=color)
    axes[1, 1].set_title("CIFAR-100: Base recorded training results")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(1, 100)
        ax.set_ylim(bottom=0, top=100)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("CLIP reproduction: recorded training results (snapshot 2026-10-22)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=180)
        fig.savefig(output.with_suffix(".svg"))
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "docs/results/training_history.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/training_curves.png")
    args = parser.parse_args()
    draw_curves(load_history(args.history), args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")
