**1. 普通图像增强（`use_timm=False`，默认路径）**

实现位置：[transform.py:440](/Users/mac/proj/open_clip/src/open_clip/transform.py:440)，参数定义在 [AugmentationCfg](/Users/mac/proj/open_clip/src/open_clip/transform.py:63)。

| 增强 | 作用 | 参数与默认状态 |
|---|---|---|
| 随机裁剪缩放 `RandomResizedCrop` | 随机选取区域，再缩放到模型输入尺寸 | 默认开启，`scale=(0.9, 1.0)`，指裁剪面积占原图的比例 |
| 颜色扰动 `color_jitter` | 随机改变亮度、对比度、饱和度、色相 | 默认关闭；需设置四元组 `color_jitter` 和执行概率 `color_jitter_prob` |
| 随机灰度化 `gray_scale` | 转为灰度图，仍保留三个通道 | 默认关闭；由 `gray_scale_prob` 控制 |

执行顺序：

```text
随机裁剪缩放 → 转 RGB → 可选颜色扰动 → 可选灰度化 → 转 Tensor → 标准化
```

这里有两个容易忽略的细节：

- 普通路径没有把 `aug_cfg.ratio` 传给 `RandomResizedCrop`，因此自定义 `ratio` 不生效，使用 torchvision 的默认宽高比范围。
- 普通路径的随机裁剪固定使用 **BICUBIC** 插值。

**2. timm 图像增强（`use_timm=True`）**

实现位置：[transform.py:410](/Users/mac/proj/open_clip/src/open_clip/transform.py:410)，由 `timm.data.create_transform()` 构建。

| 增强或控制项 | 参数 | 当前代码行为 |
|---|---|---|
| 随机裁剪缩放 | `scale` | 默认 `(0.9, 1.0)` |
| 裁剪宽高比 | `ratio` | 可以传入；未设置时使用 timm 默认值 |
| 颜色扰动 | `color_jitter` | 默认关闭，设置后交给 timm 处理 |
| 随机擦除 | `re_prob`、`re_count` | 控制执行概率和擦除区域数量；固定 `re_mode='pixel'` |
| 插值方式 | `image_interpolation` | 可传 `bicubic`、`bilinear`、`random`，交给 timm 处理 |
| 水平翻转 | `hflip` | 代码固定为 `0.`，关闭 |

**切换到 timm 后，`color_jitter_prob` 和 `gray_scale_prob` 会被移除，不起作用。** 也不能直接通过当前 `AugmentationCfg` 传入任意 timm 参数，例如 `auto_augment`。

**3. NaFlex 的尺寸变化**

入口：[NaFlexTransformFactory](/Users/mac/proj/open_clip/src/open_clip/transform.py:77)，要求同时启用 `use_timm=True naflex=True`。

除了委托 timm 构建增强，NaFlex 数据管道还支持：

- **变化的序列长度／图像 token 预算**：训练时从配置的 `seq_lens` 中采样，支持指定采样权重，见 [naflex_data.py:680](/Users/mac/proj/open_clip/src/open_clip_train/naflex_data.py:680)。
- **可变 patch 尺寸**：支持配置多个 patch 尺寸及其权重，需模型支持相应的 patch 插值。

这部分让训练输入具有不同的分辨率或 patch 粒度。

**4. 文件里还有两种未接入默认训练流程的随机变换**

[ResizeKeepRatio](/Users/mac/proj/open_clip/src/open_clip/transform.py:138) 支持：

| 随机变换 | 默认范围 | 默认执行概率 |
|---|---|---|
| 随机缩放 | `(0.85, 1.05)` | `0` |
| 随机宽高比扰动 | `(0.9, 1.11)` | `0` |

当前 `image_transform()` 调用它时没有开启这些随机选项；需要自行构建或修改 transform 才会生效。

**5. 普通预处理与验证流程**

RGB 转换、Tensor 转换、Normalize，以及下面这些验证操作属于预处理：

| 验证模式 | 操作 |
|---|---|
| `shortest`，默认 | 按短边缩放，再中心裁剪 |
| `longest` | 按长边缩放，再中心裁剪或补边 |
| `squash` | 直接缩放到目标尺寸，可能改变宽高比 |

实现见 [transform.py:467](/Users/mac/proj/open_clip/src/open_clip/transform.py:467)。普通验证流程没有随机增强。当前文本数据路径主要做 tokenization、截断和 padding，没有额外的文本增强。

**6. 仓库中的音频增强**

如果也包含当前仓库新增的音频模型，相关操作还有：

| 操作 | 作用与启用条件 |
|---|---|
| 随机时间裁剪 | 长音频随机截取固定长度，训练默认 `rand_trunc` |
| Fusion 多片段处理 | 组合前、中、后片段及一个随机截取的频谱片段，需启用 fusion |
| SpecAugment | HTSAT 训练时执行时间遮挡和频率遮挡：时间条带数 `2`、宽度参数 `64`；频率条带数 `2`、宽度参数 `8` |
| Mixup | HTSAT 中实现了频谱混合，训练时还需传入 `mixup_lambda` 才执行 |

位置：[音频预处理](/Users/mac/proj/open_clip/src/open_clip/audio/transform.py:140)、[SpecAugment 配置](/Users/mac/proj/open_clip/src/open_clip/audio/htsat.py:833)、[Mixup 调用](/Users/mac/proj/open_clip/src/open_clip/audio/htsat.py:1059)。音频重复、补零、重采样和 int16 量化属于预处理。

另外，模型内部还有 [PatchDropout](/Users/mac/proj/open_clip/src/open_clip/transformer.py:36)，训练时随机丢弃部分 patch token，属于模型正则化。

你仓库里的训练脚本有这样一组图像增强配置，见 [h14_224_32_finetune.sh](/Users/mac/proj/open_clip/scripts/h14_224_32_finetune.sh:14)：

```bash
--aug-cfg \
  scale='(0.4, 1.0)' \
  color_jitter='(0.32, 0.32, 0.32, 0.08)' \
  color_jitter_prob=0.8 \
  gray_scale_prob=0.2
```

含义是：随机裁剪原图 **40%～100% 的面积**，以 **80% 概率做颜色扰动**，以 **20% 概率做灰度化**，走普通 torchvision 路径。