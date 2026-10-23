# OpenCLIP：从零训练 CIFAR-10 / CIFAR-100

入口：[scripts/train_open_clip.py](../scripts/train_open_clip.py)。使用 OpenCLIP 3.3.0 的模型、分词器、预处理、损失和学习率调度组件，配合 CIFAR 数据适配和本项目的训练循环。图像塔、文本塔和 token embedding 均随机初始化，不加载预训练权重。

| 配置 | CIFAR-10 | CIFAR-100 |
|---|---|---|
| 图像尺寸 / Patch | 224 / 32 | 224 / 32 |
| 图像塔：层数 / 宽度 / 头数 / MLP | 12 / 768 / 12 / 3072 | 同左 |
| 文本塔：层数 / 宽度 / 头数 / MLP | 12 / 512 / 8 / 2048 | 同左 |
| 特征维度 / 词表大小 | 512 / 49408 | 同左 |
| 文本长度（含 SOT、EOT） | **8** | **16** |
| 激活函数 / Dropout | QuickGELU / **0** | 同左 |
| Batch / Epochs / Warmup epochs | 512 / 100 / 5 | 同左 |
| AdamW：LR / WD / betas / eps | 3e-4 / 0.05 / (0.9, 0.98) / 1e-6 | 同左 |
| 训练精度 / 评估精度 / Seed | BF16 / FP32 / 42 | 同左 |

模型与分词器的 context length 同时设置为 8/16，文本位置编码也只有对应的长度。模型构建显式设置 `pretrained=None`、`load_weights=False`、`pretrained_image=False`、`pretrained_text=False`、`force_quick_gelu=True` 和 `force_patch_dropout=0`；图像及文本注意力和 MLP 均无 dropout。[OpenCLIP 模型创建接口](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip/factory.py)

使用 torchvision 的 CIFAR-10/100 读取图片，通过 `create_model_and_transforms()` 获取 OpenCLIP 默认预处理。归一化均值为 `(0.48145466, 0.4578275, 0.40821073)`，标准差为 `(0.26862954, 0.26130258, 0.27577711)`；训练增强为面积 0.9～1.0 的随机裁剪并以 bicubic 放大到 224，评估使用 bicubic Resize 和 CenterCrop。变换在 DataLoader 的 CPU worker 中完成，模型训练仍在 GPU 上。这组实验采用 OpenCLIP 默认增强，与手写实验的 CIFAR 归一化、翻转、颜色扰动、灰度化和擦除设置不同。[官方预处理实现](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip/transform.py)

训练使用官方 `ClipLoss(cache_labels=True)`：每批编码全部类别文本，再按标签选择对应文本特征，计算 `[B, B]` 对称对比损失。在当前关闭 Dropout、同类共用文本特征的条件下，其批平均损失和参数梯度与原同类多正样本损失数学等价，数值计算存在浮点误差。评估使用 `[B, 10/100]` 相似度，累计所有样本（含最后不足一批的样本）的预测准确率。[官方损失实现](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip/loss.py)

AdamW 按官方规则分组：一维参数、BN/LN、bias 和 `logit_scale` 不做 WD，位置嵌入做 WD；保留上表中 CIFAR 实验使用的学习率、WD、batch 和训练轮数。`--no_logit_scale_wd` 为默认设置，添加 `--logit_scale_wd` 可开启温度参数的 WD。温度初始为 `1/0.07`，指数上限为 100。学习率由官方 `cosine_lr` 按全局训练步数执行线性 warmup 和余弦衰减。[官方优化器分组](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip_train/main.py)、[官方学习率调度](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip_train/scheduler.py)

## 启动训练

在项目根目录、已配置 PyTorch 的环境中运行：

```bash
python -m pip install -r requirements.txt

python scripts/train_open_clip.py --dataset cifar10 --n 18_open_clip \
  --save_dir ~/autodl-tmp/model/cifar-10_vit-base-224-32

python scripts/train_open_clip.py --dataset cifar100 --n 19_open_clip \
  --save_dir ~/autodl-tmp/model/cifar-100_vit-base-224-32
```

两条命令分别训练独立模型，默认从 `data/` 读取 CIFAR 数据。与 `train.py` 一致，`--n` 指定实验编号/备注，`--save_dir` 指定模型父目录，脚本自动追加 `exp{n}`。上述命令的模型目录为：

```text
~/autodl-tmp/model/cifar-10_vit-base-224-32/exp18_open_clip/
~/autodl-tmp/model/cifar-100_vit-base-224-32/exp19_open_clip/
```

TensorBoard 统一保存在项目的 `outputs/tensorboard/` 下：

```text
outputs/tensorboard/exp18_open_clip/
outputs/tensorboard/exp19_open_clip/
```

实验编号可自行调整；两个数据集使用不同的 `--n`，以区分 TensorBoard 曲线。`--data_root` 可指定其他数据目录。

## 使用 train.py 重跑手写模型

```bash
python train.py --dataset cifar10 --n 22_handwritten \
  --save_dir ~/autodl-tmp/model/cifar-10_vit-base-224-32 \
  --batch_size 512 --learning_rate 3e-4 --weight_decay 0.05

python train.py --dataset cifar100 --n 23_handwritten \
  --save_dir ~/autodl-tmp/model/cifar-100_vit-base-224-32 \
  --batch_size 512 --learning_rate 3e-4 --weight_decay 0.05
```

手写模型继续使用原来的 `train.py`。此次将默认图像预处理切换为 OpenCLIP 默认方案，文本长度自动按 CIFAR-10/100 设为 8/16；`config.py` 中图像 Dropout 设为 0。Batch、LR 和 WD 通过已有命令行参数传入，100 轮、5 轮 warmup、BF16、seed 42 沿用原默认值。

损失、优化器参数分组和学习率调度保留手写训练版本；当前 `optim.py` 仍对 `logit_scale` 做 WD。模型保存到上述父目录的 `exp22_handwritten/`、`exp23_handwritten/`，TensorBoard 保存到 `outputs/tensorboard/` 下的同名目录。续训继续使用 `train.py --resume --model_dir <实验目录>`。

## 输出与续训

每个 epoch 评估无随机增强的训练集及完整测试集，沿用现有实验按测试准确率选最佳模型的方式。

- `config.json`：模型结构、完整训练设置、类别顺序、prompts、tokens 和依赖版本。
- `metrics.csv`、`train{n}.log`：逐轮 loss、Train/Test accuracy、LR、温度和耗时；CSV 准确率为 0～1。
- `outputs/tensorboard/exp{n}/`：与手写训练相同的五个标量标签，独立于模型目录。
- `best.pt`：最佳模型权重和配置，不含优化器。
- `last.pt`：模型、优化器、全局训练步数、随机数状态及完成轮次，用于续训；恢复后按原学习率计划重建官方调度器。

续训时在原命令后添加 `--resume`，保持原始超参数和目标总轮数。例如：

```bash
python scripts/train_open_clip.py --dataset cifar100 --n 19_open_clip \
  --save_dir ~/autodl-tmp/model/cifar-100_vit-base-224-32 --resume
tensorboard --logdir outputs/tensorboard
```

续训时保持原来的 `--n` 和模型父目录 `--save_dir`，脚本从 `--save_dir/exp{n}/last.pt` 恢复。旧版使用本地预处理和调度器生成的 checkpoint 不兼容本次流程，需要新实验编号重新训练。脚本会校验训练设置，并拒绝新训练覆盖已有模型记录或混入已有 TensorBoard 曲线；仅包含 `smoke/` 的模型目录仍可启动正式训练。`--max_epochs_this_run N` 可在本次运行完成 N 轮后正常退出，保留原来的完整学习率计划，便于随后续训。

## 小规模验证

```bash
python scripts/train_open_clip.py --dataset cifar10 --n 18_open_clip \
  --save_dir ~/autodl-tmp/model/cifar-10_vit-base-224-32 --smoke_test --num_workers 2
python scripts/train_open_clip.py --dataset cifar100 --n 19_open_clip \
  --save_dir ~/autodl-tmp/model/cifar-100_vit-base-224-32 --smoke_test --num_workers 2
```

使用完整 ViT-B/32，每组仅运行 2 epochs、每轮至多两个训练 batch，warmup 为 1 epoch；训练集评估和测试集评估各取 `batch_size + 1` 张，覆盖不足一批的情况。模型输出放在 `--save_dir/exp{n}/smoke/`，TensorBoard 放在 `outputs/tensorboard/smoke/exp{n}/`。小规模验证的准确率不作为实验结果。
