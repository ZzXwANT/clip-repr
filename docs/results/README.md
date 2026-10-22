# README 实验结果快照

快照日期：2026-10-22。结果来自已有文件，本次只做解析、复制和绘图，不做模型推理。

| 文件 | 内容 |
|---|---|
| [experiment_summary.csv](experiment_summary.csv) | 15 个实验的配置、日志最佳指标、保留权重的标量指标及异常说明 |
| [training_history.csv](training_history.csv) | 1,219 条日志记录，含相对源路径、行号、时间、epoch、loss、准确率、学习率和耗时 |
| [provenance.json](provenance.json) | 保存配置、info 文本、日志 SHA-256、checkpoint epoch/best_acc、混淆矩阵文件来源和 SHA-256 |
| [training_curves.png](training_curves.png)、[SVG](training_curves.svg) | 从逐轮 CSV 重绘的曲线 |
| [tensorboard/summary.csv](tensorboard/summary.csv) | exp12 / exp13 的完整 100 轮 TensorBoard 汇总 |
| [exp12 CSV](tensorboard/exp12_dropout.csv)、[exp13 CSV](tensorboard/exp13_dropout_cifar-100.csv) | 从原始事件导出的 200 条 epoch 记录，共 1,000 个 scalar 数值 |
| [exp12 图](tensorboard/exp12_dropout.png)、[exp13 图](tensorboard/exp13_dropout_cifar-100.png) | 每组四个面板：loss、Train/Test 准确率、学习率和相似度缩放 |
| [tensorboard/provenance.json](tensorboard/provenance.json) | 原始事件与有效 CSV 的 SHA-256、数值单位、导出规则 |
| [cifar10_exp12/](cifar10_exp12/) | 原 `outputs/cifar-10_on_cifar-10/` 的 PNG 和两份 CSV 的原样副本 |
| [cifar100_to_cifar10_exp13/](cifar100_to_cifar10_exp13/) | 原 `outputs/cifar-100_on_cifar-10/` 的 PNG 和两份 CSV 的原样副本 |

源模型根目录为 `~/autodl-tmp/model/`，JSON / CSV 中的实验路径均相对此目录。`code_commit` 是整理时本地 main 的提交，不表示每个历史实验都运行于该版本。保存配置中的绝对数据根目录在 info 副本中替换为 `<project>/data`。

日志记录保留重复 epoch，不将重启或续训的同名记录直接删除。绘图脚本按 CSV 顺序保留同一实验、同一 epoch 的最后一个值；汇总表中的日志最佳值来自全部原始记录的最大值。`exp14` 包含多个重启，未绘制为连续曲线。`exp13` 原始控制台日志为空，现已用 TensorBoard 事件恢复完整 100 轮曲线；原始日志 CSV 的 1,219 行保留不变，额外的 200 行事件数据单独保存。

下载的 `outputs/tensorboard/tensorboard_exp12_dropout.csv` 和 `tensorboard_exp13_dropout_cifar-100.csv` 实际均为 AutoPanel HTML。没有把网页内容当作训练数据，也未覆盖原下载文件。有效数值来自对应运行目录下的 `events.out.tfevents.*`，五个标签为 `train/loss`、`train/accuracy`、`eval/accuracy`、`train/lr`、`train/logit_scale`。每个标签均有 epoch 1～100，按 step 对齐，保留完整精度，不做平滑。

`exp12` 的最佳 Test 为 86.03%（epoch 92，同轮 Train 96.04%），`exp13` 为 64.70%（epoch 98，同轮 Train 98.90%）。两者均与权重元数据一致。汇总 CSV 中的 `logged_*` 字段仍专指控制台日志；额外的 `tensorboard_*` 字段记录事件来源，避免把恢复数据混称为原始日志。

`checkpoint_epoch_1based` 在源码的零起始 epoch 上加 1；`checkpoint_best_pct` 是 `best_acc × 100`。这两个字段从 checkpoint 的 pickle opcode 中读取标量，没有加载模型、执行反序列化对象或读取张量存储。缺字段留空，不以 0 代替。历史只有 `acc` 的记录放在 `legacy_acc_pct`，没有自动归为 test_acc。

两个混淆矩阵均为真实标签作行、预测标签作列，各行 1,000 张，总数 10,000。准确率由 `trace(counts) / counts.sum()` 重算，分别为 86.03% 和 11.50%；按行归一化 CSV 与原始计数一致。原矩阵旁未保存完整运行命令和预处理信息，因此 JSON 明确将这些设置标为未知，不将手工记录的 11.25% 合并为同一次测量。

`exp13` 的配置中 `config.dataset = 'cifar100'` 被注释，但 info 记录训练集为 cifar100。当前评估入口缺字段时默认 cifar10，这可能影响复跑的数据归一化。本文保留现状并记录差异，不改动模型目录或已有图片。

重绘训练曲线：

```bash
python scripts/plot_training_curves.py
```

重绘 exp12 / exp13 的四面板图：

```bash
python scripts/plot_tensorboard_curves.py
```

重新从服务器目录的原始事件导出 CSV 后绘图（需要 TensorBoard）：

```bash
python scripts/plot_tensorboard_curves.py --event_root outputs/tensorboard
```

这些命令从项目根目录执行。仅重绘时使用仓库内 CSV 和 Matplotlib，不依赖 GPU、PyTorch 或源模型目录。
