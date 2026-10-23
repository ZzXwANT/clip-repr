"""Export TensorBoard scalars and plot the recorded CIFAR training curves.

Export from local events and plot:
    python scripts/plot_tensorboard_curves.py --event_root outputs/tensorboard

Redraw from the checked-in CSVs without TensorBoard or GPU:
    python scripts/plot_tensorboard_curves.py

Compare handwritten exp8/exp13 with OpenCLIP exp20/exp21:
    python scripts/plot_tensorboard_curves.py --open_clip_comparison

Draw the six README CIFAR comparison curves from TensorBoard events:
    python scripts/plot_tensorboard_curves.py --readme_training --event_root outputs/tensorboard
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
RUNS = (("exp12_dropout", "CIFAR-10"), ("exp13_dropout_cifar-100", "CIFAR-100"))
COMPARISON_RUNS = (
    ("exp8_origin-224", "CIFAR-10"), ("exp20_open_clip", "CIFAR-10"),
    ("exp13_dropout_cifar-100", "CIFAR-100"), ("exp21_open_clip", "CIFAR-100"),
)
README_TRAINING_RUNS = (
    ("exp22_handwritten", "CIFAR-10"), ("exp20_open_clip", "CIFAR-10"),
    ("exp23_handwritten", "CIFAR-100"), ("exp21_open_clip", "CIFAR-100"),
    ("exp11_more_aug", "CIFAR-10"), ("exp12_dropout", "CIFAR-10"),
    ("exp13_dropout_cifar-100", "CIFAR-100"), ("exp15_lr1e-4", "CIFAR-100"),
)
README_TRAINING_FIGURES = (
    ("CIFAR-10：手写 CLIP 与 OpenCLIP", "cifar10_handwritten_vs_openclip",
     (("exp22_handwritten", "手写 CLIP exp22", "#0072b2"),
      ("exp20_open_clip", "OpenCLIP exp20", "#d55e00"))),
    ("CIFAR-100：手写 CLIP 与 OpenCLIP", "cifar100_handwritten_vs_openclip",
     (("exp23_handwritten", "手写 CLIP exp23", "#0072b2"),
      ("exp21_open_clip", "OpenCLIP exp21", "#d55e00"))),
    ("CIFAR-10：更强图像增强", "cifar10_augmentation",
     (("exp22_handwritten", "基线 exp22", "#0072b2"),
      ("exp11_more_aug", "增强 exp11", "#d55e00"))),
    ("CIFAR-10：增强与 Dropout", "cifar10_dropout",
     (("exp22_handwritten", "基线 exp22", "#0072b2"),
      ("exp12_dropout", "增强 + Dropout exp12", "#d55e00"))),
    ("CIFAR-100：增强与 Dropout", "cifar100_augmentation_dropout",
     (("exp23_handwritten", "基线 exp23", "#0072b2"),
      ("exp13_dropout_cifar-100", "增强 + Dropout exp13", "#d55e00"))),
    ("CIFAR-100：增强、Dropout 与较小学习率", "cifar100_lower_lr",
     (("exp23_handwritten", "基线 exp23", "#0072b2"),
      ("exp15_lr1e-4", "增强 + Dropout + 较小 LR exp15", "#d55e00"))),
)
TAGS = ("train/loss", "train/accuracy", "eval/accuracy", "train/lr", "train/logit_scale")


def export_events(event_root, output_dir, runs=RUNS):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    sources = []
    for run, dataset in runs:
        run_dir = event_root / run
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard events in {run_dir}")
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0}).Reload()
        values = {}
        for tag in TAGS:
            if tag not in accumulator.Tags()["scalars"]:
                raise ValueError(f"Missing scalar {tag!r} in {run_dir}")
            values[tag] = {event.step: event for event in accumulator.Scalars(tag)}
        steps = set(values[TAGS[0]])
        if not steps or any(set(series) != steps for series in values.values()):
            raise ValueError(f"Scalar steps do not align in {run_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{run}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "wall_time", *TAGS], lineterminator="\n")
            writer.writeheader()
            for step in sorted(steps):
                writer.writerow({
                    "epoch": step, "wall_time": values["eval/accuracy"][step].wall_time,
                    **{tag: series[step].value for tag, series in values.items()},
                })
        files = [{"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                 for file in event_files]
        original_download = event_root / f"tensorboard_{run}.csv"
        download = None
        if original_download.exists():
            download = {
                "name": original_download.name,
                "sha256": hashlib.sha256(original_download.read_bytes()).hexdigest(),
                "is_html": original_download.read_text(encoding="utf-8-sig").lstrip().lower().startswith("<!doctype html"),
            }
        sources.append({
            "run": run, "dataset": dataset, "event_directory": str(run_dir),
            "event_files": files, "downloaded_csv": download,
            "exported_csv": path.name, "exported_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "epochs": len(steps), "scalar_samples": sum(len(series) for series in values.values()),
        })
    (output_dir / "provenance.json").write_text(json.dumps({
        "source": "TensorBoard scalar events; downloaded CSV files are not used.",
        "accuracy_unit": "fraction in CSV; percent in plots and summary.csv",
        "step_unit": "one epoch; training scripts write epoch + 1",
        "duplicate_step_policy": "last scalar occurrence per tag and step",
        "smoothing": "none", "runs": sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"epoch", *TAGS}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is not an exported scalar CSV; use --event_root to export events")
        rows = []
        for record in reader:
            row = {"epoch": int(record["epoch"]), **{tag: float(record[tag]) for tag in TAGS}}
            if row["epoch"] < 1 or not all(math.isfinite(row[tag]) for tag in TAGS):
                raise ValueError(f"Invalid epoch or non-finite scalar in {path}")
            if any(not 0 <= row[tag] <= 1 for tag in ("train/accuracy", "eval/accuracy")):
                raise ValueError(f"Expected accuracy fractions between 0 and 1 in {path}")
            rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    if not rows or len({row["epoch"] for row in rows}) != len(rows):
        raise ValueError(f"Empty CSV or duplicate epochs in {path}")
    return rows


def draw_run(rows, run, dataset, output_dir):
    best = max(rows, key=lambda row: row["eval/accuracy"])
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["train/loss"] for row in rows], color="#0072b2")
    axes[0, 0].set(title="Training contrastive loss", ylabel="Loss")
    axes[0, 1].plot(epochs, [100 * row["train/accuracy"] for row in rows],
                    label="Train (no augmentation)", color="#0072b2")
    axes[0, 1].plot(epochs, [100 * row["eval/accuracy"] for row in rows],
                    label="Test", color="#d55e00")
    axes[0, 1].scatter([best["epoch"]], [100 * best["eval/accuracy"]],
                       marker="*", s=120, color="#d55e00", zorder=3,
                       label=f"Best test: {100 * best['eval/accuracy']:.2f}% @ {best['epoch']}")
    axes[0, 1].set(title="Train / test accuracy", ylabel="Accuracy (%)", ylim=(0, 105))
    axes[0, 1].legend(loc="lower right", fontsize=9)
    axes[1, 0].plot(epochs, [row["train/lr"] for row in rows], color="#009e73")
    axes[1, 0].set(title="Learning rate (warmup + cosine)", ylabel="Learning rate")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 1].plot(epochs, [row["train/logit_scale"] for row in rows], color="#7b4ab5")
    axes[1, 1].set(title="Learned similarity scale", ylabel="exp(logit_scale)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_xlim(epochs[0], epochs[-1])
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset} | {run} | from scratch, no pretrained weights\n"
                 f"TensorBoard scalar events: {len(rows)} epochs, no smoothing", fontsize=13)
    try:
        fig.savefig(output_dir / f"{run}.png", dpi=180)
        fig.savefig(output_dir / f"{run}.svg")
    finally:
        plt.close(fig)


def plot_exports(input_dir, output_dir, runs=RUNS, draw_individual=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run, dataset in runs:
        rows = load_csv(input_dir / f"{run}.csv")
        if draw_individual:
            draw_run(rows, run, dataset, output_dir)
        best = max(rows, key=lambda row: row["eval/accuracy"])
        last = rows[-1]
        summary = {
            "run": run, "dataset": dataset, "epochs": len(rows), "best_epoch": best["epoch"],
            "best_test_pct": 100 * best["eval/accuracy"],
            "train_at_best_test_pct": 100 * best["train/accuracy"],
            "gap_at_best_pp": 100 * (best["train/accuracy"] - best["eval/accuracy"]),
            "last_epoch": last["epoch"], "last_train_pct": 100 * last["train/accuracy"],
            "last_test_pct": 100 * last["eval/accuracy"], "last_loss": last["train/loss"],
            "last_gap_pp": 100 * (last["train/accuracy"] - last["eval/accuracy"]),
        }
        summaries.append(summary)
        print(f"{dataset} {run}: {len(rows)} epochs; best test={summary['best_test_pct']:.2f}% "
              f"at epoch {best['epoch']}; train at best={summary['train_at_best_test_pct']:.2f}%")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def configure_chinese_font(font_path=None):
    selected = font_path.expanduser() if font_path else ROOT / ".cache/fonts/NotoSansCJKsc-Regular.otf"
    if selected.is_file():
        font_manager.fontManager.addfont(str(selected))
        name = font_manager.FontProperties(fname=str(selected)).get_name()
    else:
        name = None
        for candidate in ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
                          "SimHei", "WenQuanYi Micro Hei", "PingFang SC"):
            try:
                font_manager.findfont(candidate, fallback_to_default=False)
                name = candidate
                break
            except ValueError:
                continue
        if name is None:
            raise RuntimeError("需要中文字体；请安装 Noto Sans CJK SC，或用 --font_path 指定中文字体文件")
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False})


def draw_comparison(input_dir, output_dir):
    labels = {
        "exp8_origin-224": "手写 CLIP（exp8）",
        "exp20_open_clip": "OpenCLIP（exp20）",
        "exp13_dropout_cifar-100": "手写 CLIP（exp13）",
        "exp21_open_clip": "OpenCLIP（exp21）",
    }
    series = {run: load_csv(input_dir / f"{run}.csv") for run, _ in COMPARISON_RUNS}
    last_epoch = max(rows[-1]["epoch"] for rows in series.values())
    for dataset, filename in (("CIFAR-10", "cifar10_comparison"), ("CIFAR-100", "cifar100_comparison")):
        fig, (accuracy_ax, gap_ax) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for run, run_dataset in COMPARISON_RUNS:
            if run_dataset != dataset:
                continue
            rows = series[run]
            epochs = [row["epoch"] for row in rows]
            train = [100 * row["train/accuracy"] for row in rows]
            test = [100 * row["eval/accuracy"] for row in rows]
            color = "#d55e00" if "open_clip" in run else "#0072b2"
            label = labels[run] + f"，记录{epochs[-1]}轮"
            accuracy_ax.plot(epochs, train, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
            accuracy_ax.plot(epochs, test, color=color, linewidth=1.6, label=label)
            best = max(rows, key=lambda row: row["eval/accuracy"])
            accuracy_ax.scatter(best["epoch"], 100 * best["eval/accuracy"], color=color,
                                marker="*", s=90, zorder=3)
            accuracy_ax.annotate(f"{100 * best['eval/accuracy']:.2f}%",
                                 (best["epoch"], 100 * best["eval/accuracy"]),
                                 xytext=(-48, 9), textcoords="offset points", color=color, fontsize=10)
            gap_ax.plot(epochs, [a - b for a, b in zip(train, test)], color=color, linewidth=1.6, label=labels[run])
        accuracy_ax.set(title="训练集与测试集准确率", ylabel="准确率（%）", ylim=(0, 105))
        accuracy_ax.legend(loc="lower right", fontsize=9)
        gap_ax.set(title="训练集 − 测试集准确率差距", ylabel="差距（百分点）")
        gap_ax.legend(loc="lower right", fontsize=9)
        gap_ax.axhline(0, color="#999999", linewidth=0.7)
        for ax in (accuracy_ax, gap_ax):
            ax.set_xlabel("训练轮次")
            ax.set_xlim(1, last_epoch)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{dataset}：手写 CLIP 与 OpenCLIP 从零训练对比\n"
                     "虚线：训练集　实线：测试集　星号：最佳测试准确率；原始曲线，未平滑", fontsize=13)
        try:
            fig.savefig(output_dir / f"{filename}.png", dpi=180)
            fig.savefig(output_dir / f"{filename}.svg")
        finally:
            plt.close(fig)


def draw_readme_training(input_dir, output_dir):
    series = {run: load_csv(input_dir / f"{run}.csv") for run, _ in README_TRAINING_RUNS}
    for title, filename, runs in README_TRAINING_FIGURES:
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        for run, label, color in runs:
            rows = series[run]
            epochs = [row["epoch"] for row in rows]
            train = [100 * row["train/accuracy"] for row in rows]
            test = [100 * row["eval/accuracy"] for row in rows]
            best = max(rows, key=lambda row: row["eval/accuracy"])
            last = epochs[-1]
            record = f"；记录至 {last}/100 轮" if last < 100 else ""
            ax.plot(epochs, train, color=color, linewidth=1.3, linestyle="--",
                    alpha=0.75, label=f"{label}：Train")
            ax.plot(epochs, test, color=color, linewidth=1.8,
                    label=f"{label}：Test 最佳 {100 * best['eval/accuracy']:.2f}%"
                          f"（第 {best['epoch']} 轮）{record}")
            ax.scatter(best["epoch"], 100 * best["eval/accuracy"],
                       color=color, edgecolor="white", linewidth=0.7,
                       marker="*", s=160, zorder=3)

        ax.set(xlabel="训练轮次（epoch）", ylabel="准确率（%）", xlim=(1, 100), ylim=(0, 105))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        fig.suptitle(title + "\n虚线：Train　实线：Test　星号：最佳 Test；原始曲线，未平滑", fontsize=13)
        try:
            fig.savefig(output_dir / f"{filename}.png", dpi=180)
            fig.savefig(output_dir / f"{filename}.svg")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event_root", type=Path, help="Export raw event files before plotting")
    parser.add_argument("--open_clip_comparison", action="store_true", help="Compare exp8/20 and exp13/21")
    parser.add_argument("--readme_training", action="store_true", help="Draw the six paired CIFAR comparisons for README")
    parser.add_argument("--font_path", type=Path, help="Chinese font file for comparison plots; otherwise use an installed font or cached Noto CJK")
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    if args.open_clip_comparison and args.readme_training:
        parser.error("--open_clip_comparison and --readme_training cannot be combined")
    folder = "cifar_training" if args.readme_training else ("open_clip_from_scratch" if args.open_clip_comparison else "tensorboard")
    default_dir = ROOT / "docs/results" / folder
    input_dir = args.input_dir or default_dir
    output_dir = args.output_dir or input_dir
    runs = README_TRAINING_RUNS if args.readme_training else (COMPARISON_RUNS if args.open_clip_comparison else RUNS)
    if args.event_root is not None:
        export_events(args.event_root, input_dir, runs)
    if args.open_clip_comparison or args.readme_training:
        configure_chinese_font(args.font_path)
    plot_exports(input_dir, output_dir, runs, draw_individual=not (args.open_clip_comparison or args.readme_training))
    if args.open_clip_comparison:
        draw_comparison(input_dir, output_dir)
    if args.readme_training:
        draw_readme_training(input_dir, output_dir)
