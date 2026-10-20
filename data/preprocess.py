"""Batch preprocessing after transferring raw CIFAR images to the training device."""
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import CIFAR_STATS


class BatchImagePreprocess(nn.Module):
    """Accept uint8 BCHW images; return normalized float32 model inputs.

    Augmentation probabilities match the CPU pipeline. Color transforms run on
    the small source images before crop/resize to reduce GPU memory traffic.
    This ordering and tensor interpolation differ from the PIL CPU pipeline.
    Random crop, interpolation choice, and augmentation gates are per image.
    """

    def __init__(self, img_size, dataset_name="cifar10", aug=False):
        super().__init__()
        self.img_size = img_size
        self.aug = aug
        mean, std = CIFAR_STATS[dataset_name]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        if aug:
            try:
                import kornia.augmentation as K
                from kornia.core import ImageModule
            except ImportError as exc:
                raise ImportError(
                    "GPU augmentation requires kornia==0.8.2; "
                    "install requirements.txt or use --image_preprocess cpu"
                ) from exc
            self.crops = nn.ModuleList([
                K.RandomResizedCrop(
                    (img_size, img_size), scale=(0.4, 1.0),
                    ratio=(3 / 4, 4 / 3), resample=mode,
                    cropping_mode="resample", align_corners=True,
                    same_on_batch=False,
                )
                for mode in ("bilinear", "bicubic")
            ])
            self.color = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.ColorJitter(
                    brightness=0.32, contrast=0.32, saturation=0.32,
                    hue=0.08, p=0.8, same_on_batch=False,
                ),
                K.RandomGrayscale(p=0.2, same_on_batch=False),
            )
            self.erase = K.RandomErasing(
                p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3),
                value=0.0, same_on_batch=False,
            )
            # Kornia 0.8.2 默认为 show()/save() 缓存 output.detach().cpu()。
            # 训练不需要可视化缓存；否则每个增强都会同步并回传整批图像。
            for module in self.modules():
                if isinstance(module, ImageModule):
                    module.disable_features = True

    @torch.no_grad()
    def forward(self, images):
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected raw uint8 images with shape [B, 3, H, W]")
        images = images.float().div_(255)
        if self.aug:
            # 色相等操作包含较多中间 Tensor，在 32x32 上执行后再放大。
            images = self.color(images)
            # 只按插值方式分两组；每张图的裁剪参数仍独立采样。
            # 在 CPU 上采样分组，避免读取 CUDA 标量带来的同步。
            choice = torch.rand(images.shape[0]) < 0.5
            resized = images.new_empty((images.shape[0], 3, self.img_size, self.img_size))
            for crop, selected in zip(self.crops, (choice, ~choice)):
                indices = selected.nonzero(as_tuple=True)[0].to(images.device)
                if indices.numel():
                    resized.index_copy_(0, indices, crop(images.index_select(0, indices)))
            # 与 PIL 一样限制 bicubic 过冲后的像素范围。
            images = resized.clamp_(0, 1)
        else:
            images = F.interpolate(
                images, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        images = (images - self.mean) / self.std
        if self.aug:
            # Kornia 原生擦除仅支持常数填充；先生成单通道矩形 mask，
            # 再用标准正态噪声填充，与原 value='random' 的语义一致。
            keep = self.erase(torch.ones_like(images[:, :1])).bool()
            images = torch.where(keep, images, torch.randn_like(images))
        return images
