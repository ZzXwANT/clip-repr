# CIFAR-10 图像侧 Dropout 对照

`config.py` 中的 `image_dropout` 控制图像 Transformer 每个 block 的注意力输出、GELU 后和 MLP 输出的 dropout。默认 `0.1`；文本侧保持 `0.0`，SDPA 的注意力概率 dropout 仍为 `0.0`。调用 `model.eval()` 后所有 `nn.Dropout` 自动关闭。

进行两组从头训练的对照，分别将 `image_dropout` 设为 `0.0` 和 `0.1`，使用不同的实验名，例如：

```bash
# 先将 config.image_dropout 设置为 0.0
python train.py --n dropout0 --save_dir outputs --seed 42

# 第一组结束后，将 config.image_dropout 设置为 0.1
python train.py --n dropout01 --save_dir outputs --seed 42
```

两组保持相同的数据增强、batch size、学习率、weight decay、训练轮数和 warmup；不传 `--resume`。每组都会保存当时的 `config.py`，有效 dropout 概率也写入 `info.txt` 和训练日志。

比较无随机增强的训练集准确率、测试集准确率和两者差距。训练准确率降低不等于泛化改善，需要完整对照实验判断。

新增构造参数默认 `0.0`，旧位置参数调用可继续使用。Dropout 不引入可训练参数，线性层名称和索引保持不变，旧权重与优化器参数顺序兼容。零样本入口读取 checkpoint 同目录的配置；旧配置缺少 `image_dropout` 时按 `0.0` 处理。续训入口沿用当前训练配置，旧权重在当前默认配置下会启用图像 dropout；要复现旧训练行为应将该值设为 `0.0`。

运行相关检查：

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```
