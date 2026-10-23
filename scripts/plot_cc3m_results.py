"""Export CC3M TensorBoard scalars from the archived ZIP and draw labeled curves."""

import argparse
import csv
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "outputs/tensorboard/cc3m_tensorboard.zip"
DEFAULT_OUTPUT = ROOT / "docs/results/cc3m"
SCALAR_TAGS = (
    "train/loss",
    "val/loss",
    "val/i2t_R@1",
    "val/i2t_R@5",
    "val/i2t_R@10",
    "val/t2i_R@1",
    "val/t2i_R@5",
    "val/t2i_R@10",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_order(name):
    match = re.search(r"tfevents\.(\d+)", name)
    return int(match.group(1)) if match else 0


def load_scalars(archive):
    values = {tag: {} for tag in SCALAR_TAGS}
    sources = []
    with tempfile.TemporaryDirectory(prefix="cc3m-events-") as temp_dir:
        with zipfile.ZipFile(archive) as zipped:
            event_members = sorted(
                (item for item in zipped.infolist() if "events.out.tfevents." in item.filename),
                key=lambda item: event_order(item.filename),
            )
            for index, member in enumerate(event_members):
                local_path = Path(temp_dir) / f"{index:04d}-{Path(member.filename).name}"
                with zipped.open(member) as source, local_path.open("wb") as target:
                    target.write(source.read())
                accumulator = EventAccumulator(str(local_path), size_guidance={"scalars": 0})
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                if not tags:
                    continue
                sources.append(member.filename)
                for tag in SCALAR_TAGS:
                    if tag not in tags:
                        continue
                    for scalar in accumulator.Scalars(tag):
                        previous = values[tag].get(scalar.step)
                        if previous is None or scalar.wall_time >= previous["wall_time"]:
                            values[tag][scalar.step] = {
                                "wall_time": scalar.wall_time,
                                "value": scalar.value,
                            }
    if not values["train/loss"]:
        raise ValueError(f"No train/loss scalars found in {archive}")
    return values, sources


def configure_chinese_font(font_path=None):
    candidates = [
        Path(font_path) if font_path else None,
        ROOT / ".cache/fonts/NotoSansCJKsc-Regular.otf",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei"):
            if name in installed:
                plt.rcParams["font.family"] = name
                break
        else:
            raise RuntimeError("未找到中文字体；请用 --font_path 指定 Noto Sans CJK SC 等字体文件")
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(values, output):
    steps = sorted({step for series in values.values() for step in series})
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", *[tag.replace("/", "_").replace("@", "_recall_at_") for tag in SCALAR_TAGS]]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            row = {"step": step}
            for tag, field in zip(SCALAR_TAGS, fields[1:]):
                scalar = values[tag].get(step)
                value = scalar["value"] if scalar else ""
                if scalar and (tag.startswith("val/i2t_R@") or tag.startswith("val/t2i_R@")):
                    value = value * 100
                row[field] = value
            writer.writerow(row)
    return steps


def draw_chart(values, output, font_path=None):
    configure_chinese_font(font_path)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)

    train = values["train/loss"]
    x, y = zip(*sorted((step, item["value"]) for step, item in train.items()))
    axes[0].plot(x, y, color="#0072B2", linewidth=1.5, label="训练 Loss")
    val_loss = values["val/loss"]
    if val_loss:
        vx, vy = zip(*sorted((step, item["value"]) for step, item in val_loss.items()))
        axes[0].plot(vx, vy, color="#D55E00", marker="o", markersize=3, linewidth=1.3, label="验证 Loss")
    axes[0].set_title("训练与验证 Loss")
    axes[0].set_ylabel("Loss")

    for ax, direction, prefix, title in (
        (axes[1], "i2t", "图像到文本", "图像到文本检索（Image-to-Text）"),
        (axes[2], "t2i", "文本到图像", "文本到图像检索（Text-to-Image）"),
    ):
        colors = {1: "#0072B2", 5: "#009E73", 10: "#D55E00"}
        for k in (1, 5, 10):
            tag = f"val/{direction}_R@{k}"
            series = values[tag]
            if not series:
                continue
            sx, sy = zip(*sorted((step, item["value"] * 100) for step, item in series.items()))
            ax.plot(sx, sy, color=colors[k], marker="o", markersize=3, linewidth=1.4, label=f"{prefix} Recall@{k}")
        ax.set_title(title)
        ax.set_ylabel("Recall（%）")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("训练步数（step）")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"CC3M 训练记录（截至 {max(train):,} step）", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=180)
        fig.savefig(output.with_suffix(".svg"))
    finally:
        plt.close(fig)


def write_provenance(archive, output, sources, values):
    last_step = max(values["train/loss"])
    latest_validation = {}
    for tag in SCALAR_TAGS:
        series = values[tag]
        if tag.startswith("val/") and series:
            step = max(series)
            latest_validation[tag] = {"step": step, "value": series[step]["value"]}
    payload = {
        "archive": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),
        "archive_sha256": sha256(archive),
        "event_files_with_scalars": sources,
        "train_steps": [min(values["train/loss"]), last_step],
        "latest_train_loss": values["train/loss"][last_step]["value"],
        "validation_samples_per_evaluation": 1024,
        "latest_validation_metrics": latest_validation,
        "notes": [
            "Multiple TensorBoard event files were merged by scalar global_step.",
            "Validation retrieval metrics are computed on 1,024 recorded samples per evaluation; this is not a full validation-set result.",
            "The archive contains scalar logs only; no final training-completion status or checkpoint was inferred.",
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font_path", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    values, sources = load_scalars(archive)
    steps = write_csv(values, args.output_dir / "metrics.csv")
    draw_chart(values, args.output_dir / "cc3m_training.png", args.font_path)
    write_provenance(archive, args.output_dir / "provenance.json", sources, values)
    print(f"Exported {len(steps)} step rows through step {max(values['train/loss'])} to {args.output_dir}")


if __name__ == "__main__":
    main()
