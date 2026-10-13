from PIL import Image
from .tokenizer import TextToeknizer
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class Dataset(Dataset):
    def __init__(self, root, train, img_size, context_length, vocab_size):
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=self.transform)
        
        self.prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
        self.tokenizer = TextToeknizer(vocab_size, merges_path="/Users/mac/proj/clip-repr/model/bpe_simple_vocab_16e6.txt")
        self.context_length = context_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        text = self.prompts[label]
        tokens = self.encoder_text(text)
        
        return image, tokens, label

    def encoder_text(self, text):
        # 文本处理
        input_text = '<|startoftext|> ' + text + ' <|endoftext|>'
        _, tokens = self.tokenizer.encode(input_text)   # eg: [424, 23, 124, ...]
        
        if len(tokens) > self.context_length:
            eot_id = tokens[-1]
            tokens = tokens[:self.context_length]
            tokens[-1] = eot_id  # 截断后保留 EOT
            
        tokens += [0] * (self.context_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
        
    
    

    