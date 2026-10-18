"""Streaming reader for adjacent, same-key image/txt/json members in CC3M tar shards.

No extraction is performed. Each worker receives disjoint shards. Optional local
staging keeps at most one shard per active iterator, rather than copying all CC3M.
"""
import argparse
from contextlib import contextmanager
from io import BytesIO
import json
import logging
from pathlib import Path
import random
import shutil
import tarfile
import tempfile

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from .tokenizer import TextToeknizer

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MERGES = Path(__file__).resolve().parents[1] / "model/bpe_simple_vocab_16e6.txt"


def image_transform(img_size, train):
    resize = transforms.RandomResizedCrop(
        img_size, scale=(0.5, 1.0),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ) if train else transforms.Resize(
        img_size, interpolation=transforms.InterpolationMode.BICUBIC,
    )
    return transforms.Compose([
        resize,
        *([] if train else [transforms.CenterCrop(img_size)]),
        transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def tokenize_caption(tokenizer, caption, context_length):
    # Insert special tokens separately: literal special-token strings in captions
    # must not replace the final EOT used by encode_text's argmax pooling.
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    _, ids = tokenizer.encode(caption)
    ids = [i for i in ids if i not in (sot, eot)][:context_length - 2]
    ids = [sot, *ids, eot]
    return torch.tensor(ids + [0] * (context_length - len(ids)), dtype=torch.long)


def discover_shards(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"数据目录不存在: {root}")
    shards = sorted(root.rglob("*.tar"))
    if not shards:
        raise ValueError(f"没有找到 .tar 分片: {root}")
    return shards


def split_shards(root, val_root=None, val_shards=1, seed=42):
    shards = discover_shards(root)
    if val_root:
        train, val = shards, discover_shards(val_root)
    else:
        if not 1 <= val_shards < len(shards):
            raise ValueError("自动划分需要至少两个 tar，且 1 <= val_shards < 分片数")
        random.Random(seed).shuffle(shards)
        val, train = shards[:val_shards], shards[val_shards:]
    if set(p.resolve() for p in train) & set(p.resolve() for p in val):
        raise ValueError("训练集与验证集的 tar 分片重叠；请使用互不包含的目录")
    return train, val


def shard_manifest(shards, root):
    root = Path(root).expanduser().resolve()
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in shards]


def tar_samples(path):
    """Members of each key must be contiguous, as in img2dataset WebDataset output."""
    current_key, sample = None, {}
    with tarfile.open(path, mode="r|") as archive:
        for member in archive:
            if not member.isfile() or "." not in member.name:
                continue
            key, ext = member.name.rsplit(".", 1)
            ext = ext.lower()
            if current_key is not None and key != current_key:
                yield current_key, sample
                sample = {}
            current_key = key
            if ext not in IMAGE_EXTENSIONS | {"txt", "json"}:
                continue
            limit = 32 * 1024 * 1024 if ext in IMAGE_EXTENSIONS else 2 * 1024 * 1024
            if member.size > limit:
                sample["__error__"] = "member exceeds size limit"
                continue
            source = archive.extractfile(member)
            with source:
                sample[ext] = source.read()
        if current_key is not None:
            yield current_key, sample


def decode_sample(sample):
    if "__error__" in sample:
        raise ValueError(sample["__error__"])
    caption = sample.get("txt", b"").decode("utf-8").strip()
    if not caption and "json" in sample:
        metadata = json.loads(sample["json"].decode("utf-8"))
        caption = metadata.get("caption") or metadata.get("text") or ""
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("missing/empty caption (.txt or JSON caption/text)")
    image_bytes = next((sample[e] for e in sorted(IMAGE_EXTENSIONS) if e in sample), None)
    if image_bytes is None:
        raise ValueError("missing image")
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
    return image, caption.strip()


@contextmanager
def local_shard(source, cache_dir):
    if cache_dir is None:
        yield source
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Unique directory avoids collisions across workers, training and validation.
    with tempfile.TemporaryDirectory(prefix="cc3m-", dir=cache_dir) as temp:
        target = Path(temp) / source.name
        shutil.copyfile(source, target)
        yield target


def buffered_shuffle(source, size, rng):
    buffer = []
    for item in source:
        if len(buffer) < size:
            buffer.append(item)
        else:
            index = rng.randrange(size)
            yield buffer[index]
            buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


class CC3MDataset(IterableDataset):
    def __init__(self, shards, img_size=224, context_length=77, vocab_size=49408,
                 train=True, shuffle_buffer=512, seed=42, cache_dir=None):
        if context_length < 2 or shuffle_buffer < 0:
            raise ValueError("context_length >= 2 且 shuffle_buffer >= 0")
        self.shards = list(map(Path, shards))
        self.context_length, self.vocab_size = context_length, vocab_size
        self.train, self.shuffle_buffer, self.seed = train, shuffle_buffer, seed
        self.cache_dir = cache_dir
        self.transform = image_transform(img_size, train)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        shards = self.shards.copy()
        if self.train:
            random.Random(self.seed + self.epoch).shuffle(shards)
        shards = shards[worker_id::workers]
        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)
        if worker:
            random.seed(self.seed + self.epoch * 100003 + worker_id)
            torch.manual_seed(self.seed + self.epoch * 100003 + worker_id)
        tokenizer = TextToeknizer(self.vocab_size, MERGES)

        def raw_samples():
            for shard in shards:
                with local_shard(shard, self.cache_dir) as path:
                    for key, sample in tar_samples(path):
                        yield shard.name, key, sample

        # Buffer compressed bytes, not decoded full-resolution images or tensors.
        samples = raw_samples()
        if self.train and self.shuffle_buffer:
            samples = buffered_shuffle(samples, self.shuffle_buffer, rng)
        good = bad = 0
        try:
            for shard_name, key, sample in samples:
                try:
                    image, caption = decode_sample(sample)
                except (ValueError, TypeError, AttributeError, OSError,
                        UnidentifiedImageError, Image.DecompressionBombError) as exc:
                    bad += 1
                    if bad <= 5:
                        logging.warning("跳过 %s:%s: %s", shard_name, key, exc)
                    continue
                good += 1
                yield self.transform(image), tokenize_caption(tokenizer, caption, self.context_length)
        finally:
            logging.info("CC3M worker=%d pass=%d valid=%d skipped=%d",
                         worker_id, self.epoch, good, bad)


def inspect(root, limit):
    shards = discover_shards(root)
    print(f"Shards: {len(shards)} | total: {sum(p.stat().st_size for p in shards) / 2**30:.2f} GiB")
    valid = invalid = 0
    for key, sample in tar_samples(shards[0]):
        try:
            image, caption = decode_sample(sample)
            print(f"{key}: image={image.size} caption={caption[:160]!r}")
            valid += 1
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            print(f"{key}: INVALID {exc}")
            invalid += 1
        if valid + invalid >= limit:
            break
    if not valid:
        raise ValueError("未找到有效图文对；请检查 tar 是否按同一 key 相邻存放图片和 caption")
    print(f"Checked first shard: valid={valid}, invalid={invalid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    inspect(args.root, args.limit)
