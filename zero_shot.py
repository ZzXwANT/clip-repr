import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from config import config   # 需要导入save位置的模型参数
from data.tokenizer import TextToeknizer
from model.clip import clip


def build_dataset(data_root):
    # 沿用训练时的尺寸和归一化参数，不做随机增强。
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = datasets.ImageFolder(data_root, transform=transform)
    return dataset, dataset.classes


def build_class_tokens(class_names, template):
    tokenizer = TextToeknizer(
        config.vocab_size,
        merges_path=Path(__file__).resolve().parent / "model" / "bpe_simple_vocab_16e6.txt",
    )
    tokens_list = []
    for name in class_names:
        prompt = template.format(name.replace("_", " "))
        # 与 data/dataset.py 的编码、截断和 padding 保持一致。
        _, tokens = tokenizer.encode('<|startoftext|> ' + prompt + ' <|endoftext|>')
        if len(tokens) > config.context_length:
            eot_id = tokens[-1]
            tokens = tokens[:config.context_length]
            tokens[-1] = eot_id
        tokens += [0] * (config.context_length - len(tokens))
        tokens_list.append(torch.tensor(tokens, dtype=torch.long))
    return torch.stack(tokens_list)


def load_model(checkpoint_path, device):
    # checkpoint 还包含优化器等状态；只保留推理所需权重。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    del checkpoint  # 减少资源占用
    model = clip(
        config.img_size, config.patch_size, config.in_chans,
        config.embed_dims_i, config.embed_dims_t, config.num_heads_i,
        config.num_heads_t, config.hidden_dims_i, config.hidden_dims_t, config.num_layers,
        config.output_dims, config.context_length, config.vocab_size,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, zero_shot_loader, device, class_tokens, topk=1):
    model.eval()
    class_features = model.encode_text(class_tokens.to(device))
    scale = model.logit_scale.exp()
    correct = 0
    total = 0

    for batch in tqdm(zero_shot_loader, desc="Zero-shot"):
        # 兼容 (image, label) 和现有 Dataset 的 (image, tokens, label)。
        imgs, labels = batch[0], batch[-1]
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        img_features = model.encode_image(imgs)
        logits = img_features @ class_features.T * scale
        preds = logits.topk(topk, dim=1).indices
        correct += (preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, class_names = build_dataset(args.data_root)
    class_tokens = build_class_tokens(class_names, args.template)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    model = load_model(args.checkpoint, device)
    print(f"Device: {device} | classes: {len(class_names)} | images: {len(dataset)}")
    acc = evaluate(model, loader, device, class_tokens, topk=args.topk)
    print(f"Top-{args.topk} accuracy: {acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="train.py 保存的 best.pt 路径")
    parser.add_argument("--data_root", type=Path, required=True, help="评估图片目录：根目录/类别名/图片")
    parser.add_argument("--template", default="a photo of a {}", help="类别提示词模板，使用 {} 填入类别名")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=1, help="计算 Top-k 准确率，k 不得超过类别数")
    args = parser.parse_args()

    main(args)
