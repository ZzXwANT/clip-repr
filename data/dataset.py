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
    def __init__(self, root, train, img_size, context_length, vocab_size):
        # 数据增强
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=self.transform_train if train else self.transform_test)
        
        self.prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
        self.tokenizer = TextToeknizer(vocab_size, merges_path = merges_path)
        self.context_length = context_length
        # 由于类别与tokenizer不变，提前缓存
        self.tokens_list = torch.stack([
            self.encode_text(text)
            for text in self.prompts
        ])  # [10, 77]，torch.long
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        
        return image, self.tokens_list[label], label

    def encode_text(self, text):
        # 文本处理
        input_text = '<|startoftext|> ' + text + ' <|endoftext|>'
        _, tokens = self.tokenizer.encode(input_text)   # eg: [424, 23, 124, ...]
        
        if len(tokens) > self.context_length:
            eot_id = tokens[-1]
            tokens = tokens[:self.context_length]
            tokens[-1] = eot_id  # 截断后保留 EOT
            
        tokens += [0] * (self.context_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
        
    
    

    