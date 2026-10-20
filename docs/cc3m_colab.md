# CC3M / Colab A100

本分支保留 `train.py` 的 CIFAR-10/100 入口，使用 `train_cc3m.py` 从头训练现有 ViT-B/32 风格的 CLIP。默认分辨率 224、文本长度 77、每 512 条 caption 批量分词、batch 内双向配对交叉熵。新实验不会自动加载旧权重。

已合入 main 的模型、dropout 和 AdamW 更新。新训练默认读取 `config.py` 的 `image_dropout=0.1`，也可通过 `--image-dropout` 覆盖；AdamW 使用 betas=(0.9, 0.98)、eps=1e-6，沿用当前 main 对 logit_scale 施加 weight decay 的设置。CIFAR 的 GPU 增强仍由 `train.py` 使用；CC3M 保留自身的裁剪和 CLIP 归一化。

## 不推送 GitHub 也能使用

在本地项目根目录执行：

```bash
python scripts/export_colab.py --output /tmp/clip-repr-colab.zip
```

输出 `/tmp/clip-repr-colab.zip`，包含 Git 已跟踪文件及未被忽略的新文件，并使用文件当前内容，因此未提交的修改也会打包。不会包含被忽略的数据、outputs 或 `.git`。将此 ZIP 从当前机器下载到电脑。

把 `notebooks/cc3m_colab.ipynb` 上传到 Colab，选择 A100，然后按顺序运行。更新 Drive 中的 `clip-repr` 项目目录；若该目录没有训练入口，Notebook 会要求上传上述 ZIP 并使用解压后的本地代码。依赖文件 `requirements-colab.txt` 沿用 Colab 自带的 CUDA PyTorch/torchvision。

## 数据格式

`--data-root` 指向包含 `.tar` 分片的目录，支持递归查找。同一个样本的成员必须相邻，且使用相同 key，例如：

```text
000000001.jpg
000000001.txt
000000002.png
000000002.json
```

支持 jpg/jpeg/png/webp；caption 优先使用 UTF-8 `.txt`，否则读取 `.json` 的 `caption` 或 `text` 字段。图片与 caption 分别集中存放在 tar 两个区域、仅有 URL 标注、或其他格式需要另行适配。单张压缩图片限制 32 MiB，单条文本或 JSON 限制 2 MiB，超限样本跳过。

先检查首个分片的实际内容：

```bash
python -m data.cc3m /content/drive/MyDrive/cc3m --limit 10
```

默认从全部分片中按固定 seed 留出一个验证分片，至少需要两个 tar。可用 `--val-shards` 增加留出数量；也可用 `--val-root` 指定独立验证目录。训练目录和验证目录不能互相包含，否则递归扫描会产生重叠并报错。按分片划分能避免同一个文件同时用于训练和验证，但不会自动检测跨分片的重复图片。

坏图片、缺失 caption 等样本会跳过，每个 worker 的前五条错误会输出原因，并在迭代关闭时记录 valid/skipped 计数。tar 损坏、Drive 读取或复制失败会报错，避免静默丢弃整个分片。

## 训练

Notebook 沿用提供的路径和训练参数，默认使用新实验名 `cc3m_a100_02`、`RESUME=False`。先跑独立的 100 步短测，观察有限 loss、峰值显存和 samples/s，再开始正式训练。

等价命令：

```bash
python train_cc3m.py \
  --data-root /content/drive/MyDrive/datasets/cc3m \
  --val-root /content/drive/MyDrive/datasets/cc3m_val \
  --save-dir /content/runs/cc3m_a100_02 \
  --drive-save-dir /content/drive/MyDrive/clip-repr-runs/cc3m_a100_02 \
  --cache-dir /content/cc3m-cache \
  --device cuda --precision bf16 --context-length 77 \
  --batch-size 512 --num-workers 10 \
  --learning-rate 3e-4 --weight-decay 0.1 \
  --total-steps 30000 --warmup-steps 1000 \
  --save-every 500 --eval-every 1000 --val-samples 1024
```

Notebook 使用已跑过的物理 batch=512、workers=10；命令行入口仍允许覆盖这些参数。每步负样本集合来自当前 batch；普通梯度累积不等价于扩大这个集合。启动日志显示实际 dropout 和 AdamW 参数，训练日志显示学习率、峰值显存和吞吐量，TensorBoard 另记录梯度范数并同步到 Drive。

每次迭代遍历分配给各 worker 的分片，打乱分片和压缩样本缓冲；遍历结束后重新打乱继续，直到达到 `total-steps`。它不是 epoch 数，也不根据名义上的 CC3M 样本数推测步数。每个 worker 丢弃末尾不足一个 batch 的样本，batch 至少为 2。

`--cache-dir` 将每个活跃迭代器当前读取的 tar 复制到本地临时目录，使用后清理。训练的多个 worker 加上验证迭代器会同时占用空间；预留这些 tar 及模型、优化器、last/best 和临时 checkpoint 的空间。强制杀进程可能遗留缓存，重建 Colab 运行时可以清理。

Google 建议减少 Drive 的小文件读写，将归档复制到 VM 本地读取。这里缓存完整 tar 并流式读取，不解压全部图片：[Colab FAQ](https://research.google.com/colaboratory/faq.html)。

## 保存与续训

`last.pt` 包含模型、优化器、调度器、全局步数、配置、分片清单及随机状态；验证池损失更低时更新 `best.pt`。每次保存同步 checkpoint、日志和 TensorBoard 到 Drive；配置写入 `config.py`、`run.json`、`shards.json`。本地及 Drive 文件均先写临时文件再重命名；Drive 挂载层重命名不代表云端持久化已完成。

恢复命令使用相同参数并追加：

```bash
--resume /content/drive/MyDrive/clip-repr-runs/cc3m_a100_01/last.pt
```

Notebook 中则将 `RESUME=True`。实验名、模型、分片相对路径与大小、batch、总步数、warmup、seed、precision、shuffle buffer、学习率、weight decay 和验证池大小需要保持一致。分片清单检查不计算内容哈希，不检测同大小文件内容被替换的情况。

恢复优化器和学习率进度后，数据从下一次分片打乱开始；不恢复 worker 预取队列与样本游标，部分样本可能重复。已有 CIFAR-10 checkpoint 的文本位置编码长度不同，也缺少本入口的恢复元数据，不能直接作为 CC3M 断点。

旧 CC3M checkpoint 可继续恢复：缺失的 dropout 按原值 0 处理，保留旧版 logit_scale 参数分组以及保存的 AdamW 超参数。换代码后续训不会自动改成新的优化器配置；要验证当前 main 的配置，请使用新实验名从头训练。

可以通过 `--stop-after-steps 10` 限制一次会话的更新次数，仍保留完整 `total-steps` 学习率计划。到达会话限制时评估并保存。中断或崩溃后恢复最近成功保存的 checkpoint，最多损失最近一次保存之后的训练。

## 评估与验证

验证使用固定分片顺序、固定前缀候选池（默认最多 1024 个有效图文对）。两个方向分别对整个池计算 loss 与 Recall@1/5/10，日志记录实际候选数；不同候选数的指标不可直接比较，也不代表在整个 CC3M 上的检索结果。该评估假设索引一一配对，不额外合并重复 caption 的正样本。

`zero_shot.py` 支持 CIFAR-10/100 测试集。CC3M 保存的 `config.py` 标记了 CLIP 预处理；CIFAR 模型继续使用其训练数据的归一化。切换目标数据集不改变模型的归一化：

```bash
python zero_shot.py \
  --dataset cifar100 \
  --checkpoint /content/drive/MyDrive/clip-repr-runs/cc3m_a100_02/best.pt \
  --data_root /content/data
```

本地合成数据验证：

分词优化只需要更新 `data/tokenizer.py` 和 `data/cc3m.py`。各 worker 在 CPU 上按最多 512 条 caption 批量生成 token 张量，仍逐条输出图文对供原 DataLoader 组 batch。分词批量大小不改变模型的 batch size；不需要修改训练命令或模型权重。

BPE 使用每个 worker 最多 32768 个普通词的 LRU 缓存，批内重复 caption 只编码一次。token ID、SOT/EOT、截断和 padding 规则与旧版一致，现有 checkpoint 可继续使用。文本批处理阶段只额外缓冲压缩样本，图像仍逐条解码，不缓存 512 张完整图片。此优化不涉及 tar 异步预取，也不减少文本编码器的计算量。

将两个文件替换到 Colab 项目后，需从已保存的 checkpoint 重新启动训练进程才会生效；不需要重新下载数据或重新安装依赖。不要在训练进程及其 worker 仍然运行时替换这两个文件，避免不同 worker 加载到不同版本。

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

覆盖 tar 图文配对、文本截断与 EOT、多进程分片去重、缓存异常清理、配对损失、旧 CIFAR 调用、全候选池检索、训练保存及续训。真实 Drive 分片格式和 Colab A100 性能还需在 Notebook 中验证。
