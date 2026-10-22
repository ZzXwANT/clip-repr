"""Export the exp12/exp13 scalar events and plot their recorded training curves.

Export from local events and plot:
    python scripts/plot_tensorboard_curves.py --event_root outputs/tensorboard

Redraw from the checked-in CSVs without TensorBoard or GPU:
    python scripts/plot_tensorboard_curves.py
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


ROOT = Path(__file__).resolve().parents[1]
RUNS = (("exp12_dropout", "CIFAR-10"), ("exp13_dropout_cifar-100", "CIFAR-100"))
TAGS = ("train/loss", "train/accuracy", "eval/accuracy", "train/lr", "train/logit_scale")


def export_events(event_root, output_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    sources = []
    for run, dataset in RUNS:
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
            writer = csv.DictWriter(stream, fieldnames=["epoch", "wall_time", *TAGS])
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
        "source": "TensorBoard scalar events; the downloaded .csv files contain HTML, not training data.",
        "accuracy_unit": "fraction in CSV; percent in plots and summary.csv",
        "step_unit": "one epoch; train.py writes epoch + 1",
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


def plot_exports(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run, dataset in RUNS:
        rows = load_csv(input_dir / f"{run}.csv")
        draw_run(rows, run, dataset, output_dir)
        best = max(rows, key=lambda row: row["eval/accuracy"])
        last = rows[-1]
        summary = {
            "run": run, "dataset": dataset, "epochs": len(rows), "best_epoch": best["epoch"],
            "best_test_pct": 100 * best["eval/accuracy"],
            "train_at_best_test_pct": 100 * best["train/accuracy"],
            "last_epoch": last["epoch"], "last_train_pct": 100 * last["train/accuracy"],
            "last_test_pct": 100 * last["eval/accuracy"], "last_loss": last["train/loss"],
        }
        summaries.append(summary)
        print(f"{dataset} {run}: {len(rows)} epochs; best test={summary['best_test_pct']:.2f}% "
              f"at epoch {best['epoch']}; train at best={summary['train_at_best_test_pct']:.2f}%")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event_root", type=Path, help="Export raw event files before plotting")
    parser.add_argument("--input_dir", type=Path, default=ROOT / "docs/results/tensorboard")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "docs/results/tensorboard")
    args = parser.parse_args()
    if args.event_root is not None:
        export_events(args.event_root, args.input_dir)
    plot_exports(args.input_dir, args.output_dir)
