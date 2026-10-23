# CIFAR 从零训练：README 曲线数据

本目录对应 README 第 2 节的六张成对训练曲线：两张手写 CLIP 与 OpenCLIP 的同数据集对比，以及四张改进实验与初始手写配置的对比。CIFAR-10 固定以 `exp22` 为基线，CIFAR-100 固定以 `exp23` 为基线。模型均从随机初始化开始训练；六张图尺寸及坐标范围一致。每组蓝色为基线（在实现对比图中为手写 CLIP），橙色为对照或改进。

- [CIFAR-10 手写 exp22 vs OpenCLIP exp20](cifar10_handwritten_vs_openclip.png) · [SVG](cifar10_handwritten_vs_openclip.svg)
- [CIFAR-100 手写 exp23 vs OpenCLIP exp21](cifar100_handwritten_vs_openclip.png) · [SVG](cifar100_handwritten_vs_openclip.svg)
- [CIFAR-10 增强：exp22 vs exp11](cifar10_augmentation.png) · [SVG](cifar10_augmentation.svg)
- [CIFAR-10 增强与 Dropout：exp22 vs exp12](cifar10_dropout.png) · [SVG](cifar10_dropout.svg)
- [CIFAR-100 增强与 Dropout：exp23 vs exp13](cifar100_augmentation_dropout.png) · [SVG](cifar100_augmentation_dropout.svg)
- [CIFAR-100 增强、Dropout 与较小 LR：exp23 vs exp15](cifar100_lower_lr.png) · [SVG](cifar100_lower_lr.svg)

数据由 `outputs/tensorboard/` 中八组实验的原始事件导出为同名 CSV。CSV 中 Accuracy 为 0～1，图中换算为百分比；图上的星号与图例标出每组最佳 Test Accuracy 及其 epoch。Train 曲线来自无随机增强的训练集评估。原始曲线不做平滑，重复 step 取最后一条记录。[summary.csv](summary.csv) 汇总最佳轮次，[provenance.json](provenance.json) 保存源事件和导出文件的 SHA-256。`exp22/23` 使用 OpenCLIP 图像预处理，后续实验使用手写增强；改进图比较的是相对于初始配置的累计变化。

`exp15` 的 CSV 合并原始 1～72 轮与续训的 73～100 轮；`exp21` 仅有 1～93 轮记录。此前的对照归档未用于本次图表。

从项目根目录用已导出的 CSV 重绘：

```bash
python scripts/plot_tensorboard_curves.py --readme_training
```

需要从原始 TensorBoard 事件重新导出时：

```bash
python scripts/plot_tensorboard_curves.py --readme_training --event_root outputs/tensorboard
```

中文图表需有可用中文字体；脚本会优先使用 `.cache/fonts/NotoSansCJKsc-Regular.otf`，也可通过 `--font_path` 指定字体文件。
