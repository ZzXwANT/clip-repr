"""Synthetic tar tests; no CC3M download or external services required."""
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import config
from data.cc3m import (CC3MDataset, decode_sample, local_shard, split_shards,
                       tar_samples, tokenize_caption)
from data.tokenizer import TextToeknizer
from loss import contrastive_loss, paired_contrastive_loss
from model.clip import clip
from train_cc3m import evaluate

ROOT = Path(__file__).resolve().parents[1]
SMALL_CONFIG = dict(
    img_size=16, patch_size=8, in_chans=3, embed_dims_i=16, embed_dims_t=16,
    num_heads_i=2, num_heads_t=2, hidden_dims_i=32, hidden_dims_t=32,
    num_layers=1, output_dims=8, context_length=12, vocab_size=config.vocab_size,
)


def make_shards(root):
    root.mkdir(parents=True, exist_ok=True)
    for shard_id in range(4):
        with tarfile.open(root / f"{shard_id:05d}.tar", "w") as archive:
            for index in range(8):
                key = f"{shard_id:05d}{index:03d}"
                image_data = BytesIO()
                Image.new("RGB", (24, 20), (index * 25, shard_id * 50, 80)).save(image_data, "PNG")
                caption = f"a photo of object {shard_id} number {index}"
                fields = {"png": image_data.getvalue()}
                if index % 2:
                    fields["json"] = json.dumps({"caption": caption}).encode()
                else:
                    fields["txt"] = caption.encode()
                for ext, content in fields.items():
                    member = tarfile.TarInfo(f"{key}.{ext}")
                    member.size = len(content)
                    archive.addfile(member, BytesIO(content))
            for ext, content in {"jpg": b"broken", "txt": b"invalid image"}.items():
                member = tarfile.TarInfo(f"bad.{ext}")
                member.size = len(content)
                archive.addfile(member, BytesIO(content))


class CC3MTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        make_shards(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def test_txt_json_and_broken_images(self):
        samples = list(tar_samples(self.data / "00000.tar"))
        self.assertEqual(len(samples), 9)
        for index, (_, sample) in enumerate(samples[:8]):
            image, caption = decode_sample(sample)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(caption, f"a photo of object 0 number {index}")
        with self.assertRaises(OSError):
            decode_sample(samples[-1][1])

    def test_split_repeatable_and_disjoint(self):
        train, val = split_shards(self.data)
        self.assertEqual((train, val), split_shards(self.data))
        self.assertFalse(set(train) & set(val))
        with self.assertRaises(ValueError):
            split_shards(self.data, self.data)

    def test_multiworker_does_not_duplicate_samples(self):
        shards = sorted(self.data.glob("*.tar"))
        ds = CC3MDataset(shards, img_size=16, train=False)
        single = list(DataLoader(ds, batch_size=None, num_workers=0))
        multi = list(DataLoader(ds, batch_size=None, num_workers=2))
        self.assertEqual(len(single), 32)
        self.assertEqual(len(multi), 32)
        self.assertEqual(sorted(t.tolist() for _, t in single), sorted(t.tolist() for _, t in multi))

    def test_cache_cleanup_on_error(self):
        cache = self.root / "cache"
        with self.assertRaises(RuntimeError):
            with local_shard(self.data / "00000.tar", cache) as staged:
                self.assertTrue(staged.exists())
                self.assertEqual(staged.read_bytes(), (self.data / "00000.tar").read_bytes())
                raise RuntimeError("simulate interrupted reading")
        self.assertEqual(list(cache.iterdir()), [])

    def test_caption_preserves_eot_when_truncated(self):
        tokenizer = TextToeknizer(config.vocab_size, ROOT / "model/bpe_simple_vocab_16e6.txt")
        tokens = tokenize_caption(tokenizer, "hello " * 100 + "<|endoftext|>", 8)
        self.assertEqual(tokens.shape, (8,))
        self.assertEqual(tokens[-1], tokenizer.encoder["<|endoftext|>"])
        self.assertEqual(tokens.argmax().item(), 7)

    def test_pair_loss_and_legacy_backward(self):
        logits = (torch.eye(4) * 10).requires_grad_()
        self.assertLess(paired_contrastive_loss(logits).item(), .001)
        permuted = logits[:, [1, 0, 2, 3]]
        self.assertGreater(paired_contrastive_loss(permuted).item(), 1)
        paired_contrastive_loss(logits).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        model = clip(**SMALL_CONFIG)
        images = torch.randn(4, 3, 16, 16)
        tokens = torch.zeros(4, 12, dtype=torch.long)
        tokens[:, 0] = config.vocab_size - 1
        self.assertEqual(model(images, tokens).shape, (4, 4))
        legacy_logits = model(images, tokens, torch.tensor([0, 0, 1, 1]), tokens[:2])
        contrastive_loss(legacy_logits, torch.tensor([0, 0, 1, 1])).backward()

    def test_retrieval_uses_global_validation_pool(self):
        class IdentityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

            def encode_image(self, images):
                return images

            def encode_text(self, tokens):
                return tokens

        features = torch.eye(12)
        loader = DataLoader(list(zip(features, features)), batch_size=4)
        metrics = evaluate(IdentityModel(), loader, torch.device("cpu"), "fp32", 12)
        self.assertEqual(metrics["samples"], 12)
        self.assertEqual(metrics["i2t_R@1"], 1)
        self.assertEqual(metrics["t2i_R@1"], 1)
        expected = paired_contrastive_loss(features * torch.exp(torch.tensor(1.0))).item()
        self.assertAlmostEqual(metrics["loss"], expected, places=5)

    def test_train_save_sync_and_resume(self):
        cfg = self.root / "tiny.py"
        cfg.write_text("class config:\n" + "".join(f"    {k} = {v!r}\n" for k, v in SMALL_CONFIG.items()))
        save, drive = self.root / "run", self.root / "drive"
        command = [sys.executable, str(ROOT / "train_cc3m.py"),
                   "--data-root", str(self.data), "--save-dir", str(save),
                   "--drive-save-dir", str(drive), "--model-config", str(cfg),
                   "--context-length", "12", "--device", "cpu", "--precision", "fp32",
                   "--batch-size", "4", "--num-workers", "2", "--total-steps", "3",
                   "--warmup-steps", "1", "--save-every", "1", "--eval-every", "1",
                   "--log-every", "1", "--val-samples", "6", "--shuffle-buffer", "4"]
        env = {**os.environ, "OMP_NUM_THREADS": "2"}
        first = subprocess.run(command + ["--stop-after-steps", "2"], cwd=ROOT,
                               env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(first.returncode, 0, first.stderr)
        state = torch.load(drive / "last.pt", weights_only=True)
        self.assertEqual(state["global_step"], 2)
        before = state["model"]["img_encoder.PatchEmbedding.proj.weight"].clone()
        second = subprocess.run(command + ["--resume", str(drive / "last.pt")], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(second.returncode, 0, second.stderr)
        state = torch.load(save / "last.pt", weights_only=True)
        self.assertEqual(state["global_step"], 3)
        self.assertEqual(state["scheduler"]["last_epoch"], 3)
        self.assertFalse(torch.equal(before, state["model"]["img_encoder.PatchEmbedding.proj.weight"]))
        self.assertTrue((drive / "best.pt").exists())
        self.assertTrue((drive / "config.py").exists())
        self.assertTrue(list((drive / "tensorboard").glob("events.*")))
        mismatch = subprocess.run(command + ["--resume", str(save / "last.pt"), "--batch-size", "2"],
                                  cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("恢复时", mismatch.stderr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
