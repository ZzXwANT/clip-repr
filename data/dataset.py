from PIL import Image
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

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class Dataset(Dataset):
    def __init__(self, root, train: bool, aug: bool, img_size, context_length, vocab_size):
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=self.transform_aug if aug else self.transform)
        
        self.prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
        self.tokenizer = TextToeknizer(vocab_size, merges_path = merges_path)
        self.context_length = context_length
        # 由于类别与tokenizer不变，提前缓存
        self.class_tokens = torch.stack([
            self.preprocess_text(text)
            for text in self.prompts
        ])  # [10, 77]，torch.long
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
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
