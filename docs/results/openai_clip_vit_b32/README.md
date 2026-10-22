# OpenAI CLIP ViT-B/32 零样本评估

使用 OpenAI 官方预训练权重、官方 Python 包的 `clip.load(..., jit=False)` 图像预处理及 `clip.tokenize`（长度 77）。每个类别仅使用一个提示词 `a photo of a {}`；保留 torchvision 类别顺序，将 CIFAR-100 类名中的下划线替换为空格。不训练、不微调、不做 prompt ensemble。

| 完整测试集 | 正确数 / 总数 | Top-1 | 实测记录 | 混淆矩阵 |
|---|---:|---:|---|---|
| CIFAR-10 | 8880 / 10000 | **88.80%** | [metrics.json](cifar10/metrics.json) | [PNG](cifar10/confusion_matrix.png) · [计数](cifar10/confusion_matrix.csv) · [归一化](cifar10/confusion_matrix_normalized.csv) |
| CIFAR-100 | 6415 / 10000 | **64.15%** | [metrics.json](cifar100/metrics.json) | [PNG](cifar100/confusion_matrix.png) · [计数](cifar100/confusion_matrix.csv) · [归一化](cifar100/confusion_matrix_normalized.csv) |

图片预处理为 bicubic Resize 到 224、CenterCrop 224、转 RGB、ToTensor，再使用 CLIP 均值 `(0.48145466, 0.4578275, 0.40821073)` 和标准差 `(0.26862954, 0.26130258, 0.27577711)`。图文编码后转 FP32 并做 L2 归一化，以相似度最大值作为预测；CUDA 图文编码使用官方加载器的 FP16 权重，FP32 相似度计算关闭 TF32。

权重位于 `~/autodl-tmp/model/ViT-B-32.pt`，SHA-256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`，与 [OpenAI 官方下载地址](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) 一致。权重不提交到 Git。

从项目根目录运行：

```bash
conda activate clip
python -m pip install -r requirements.txt
OMP_NUM_THREADS=8 python scripts/evaluate_openai_clip.py --device cuda
```

可用 `--dataset cifar10` 或 `--dataset cifar100` 单独评估，`--checkpoint` 指定官方权重路径，`--data-root` 指定数据位置。脚本只读取本地数据和权重，不自动下载。无 GPU 时可运行 `--device cpu --smoke-test`，仅处理每个测试集前 8 张图片，结果保存在 `smoke/` 子目录，不能作为测试集准确率。

每个数据集目录保存：

| 文件 | 内容 |
|---|---|
| `metrics.json` | Top-1、正确数/样本数、类别顺序、全部提示词、权重/数据/脚本/输出文件 SHA-256、依赖版本、设备、运行命令和推理耗时 |
| `predictions.csv` | 按原测试集顺序记录图片索引、真实标签、预测标签；数字标签对应 metrics 中的 classes |
| `confusion_matrix.csv` | 真实类别作行、预测类别作列的计数矩阵 |
| `confusion_matrix_normalized.csv` | 按真实类别归一化的比例矩阵 |
| `confusion_matrix.png` | 计数和按行归一化两个面板；CIFAR-100 数值见 CSV |

脚本检查矩阵总数、逐类样本数、对角线正确数与逐图预测一致。Top-1 为 `trace(counts) / counts.sum()`，百分比为该比例乘以 100。

与仓库模型比较时，官方模型使用大规模图文预训练，仓库模型则从随机初始化开始使用 CIFAR 类别监督训练，且按测试集选择最佳轮次；预处理也不同。这是两种训练来源和评估流程的实测对照。

实现来源：[OpenAI CLIP](https://github.com/openai/CLIP)，包版本固定在 [d05afc4](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)。
