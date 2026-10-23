from .tokenizer import TextToeknizer
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
from pathlib import Path

merges_path = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "bpe_simple_vocab_16e6.txt"
)

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

class Dataset(Dataset):
    def __init__(self, root, train: bool, aug: bool, img_size, context_length, vocab_size,
                 dataset_name="cifar10", class_tokens=None, raw_images=False, image_transform=None):
        self.dataset_name = dataset_name
        self.raw_images = raw_images
        mean, std = CIFAR_STATS[dataset_name]
        # 数据增强
        self.transform_aug = transforms.Compose(
            [
                # 随机裁剪面积、宽高比和插值方式，输出尺寸保持固定。
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(
                        (img_size, img_size),
                        scale=(0.4, 1.0),
                        ratio=(3 / 4, 4 / 3),
                        interpolation=interpolation,
                    )
                    for interpolation in (
                        transforms.InterpolationMode.BILINEAR,
                        transforms.InterpolationMode.BICUBIC,
                    )
                ]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08,
                    ),
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # 在标准化后的 Tensor 上用随机像素擦除一个区域。
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random',
                ),
            ]
        )
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform = image_transform if image_transform is not None else (self.transform_aug if aug else self.transform)
        dataset_class = datasets.CIFAR100 if dataset_name == "cifar100" else datasets.CIFAR10
        self.ds = dataset_class(
            root=root, train=train, download=True,
            transform=None if raw_images else self.transform,
        )
        self.classes = list(self.ds.classes)
        self.prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.classes]
        self.tokenizer = TextToeknizer(vocab_size, merges_path = merges_path)
        self.context_length = context_length
        # 由于类别与tokenizer不变，提前缓存
        if class_tokens is None:
            self.class_tokens = torch.stack([
                self.preprocess_text(text)
                for text in self.prompts
            ])
        else:
            self.class_tokens = class_tokens
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.raw_images:
            # 直接读取 CIFAR 的内存数组，跳过 PIL 和逐图 CPU 增强。
            # DataLoader 合并为 [B, 3, 32, 32] uint8，交给主进程在 GPU 处理。
            image = torch.from_numpy(self.ds.data[idx]).permute(2, 0, 1)
            label = self.ds.targets[idx]
        else:
            image, label = self.ds[idx]
        
        return image, self.class_tokens[label], label

    def preprocess_text(self, text):
        # 文本处理
        input_text = '<|startoftext|> ' + text + ' <|endoftext|>'
        _, tokens = self.tokenizer.encode(input_text)   # eg: [424, 23, 124, ...]
        
        if len(tokens) > self.context_length:
            eot_id = tokens[-1]
            tokens = tokens[:self.context_length]
            tokens[-1] = eot_id  # 截断后保留 EOT
            
        tokens += [0] * (self.context_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
