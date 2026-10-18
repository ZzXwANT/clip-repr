"""Dropout behavior and compatibility checks using small synthetic models."""
from contextlib import ExitStack
import copy
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from config import config
from model.clip import clip
from model.GELU import GELU
from model.LayerNorm import LayerNorm
from model.MHA import MultiHeadAtten
from model.ResAttenBlock import ResAttenBlock
from optim import create_optimizer
import zero_shot

ROOT = Path(__file__).resolve().parents[1]
SMALL_CONFIG = dict(
    img_size=16, patch_size=8, in_chans=3, embed_dims_i=16, embed_dims_t=16,
    num_heads_i=2, num_heads_t=2, hidden_dims_i=32, hidden_dims_t=32,
    num_layers=2, output_dims=8, context_length=8, vocab_size=32,
)


class LegacyBlock(nn.Module):
    """The block architecture before dropout, including its parameter names."""
    def __init__(self, mask=False):
        super().__init__()
        self.atten = MultiHeadAtten(16, 2, mask)
        self.ln1 = LayerNorm(16)
        self.ln2 = LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 32), GELU(), nn.Linear(32, 16))

    def forward(self, x):
        x = x + self.atten(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SyntheticCIFAR(Dataset):
    def __init__(self, *args, **kwargs):
        generator = torch.Generator().manual_seed(42)
        self.images = torch.randn(6, 3, 16, 16, generator=generator)
        self.labels = torch.arange(6) % 3
        self.class_tokens = torch.zeros(3, 8, dtype=torch.long)
        self.class_tokens[:, 0] = 30
        self.class_tokens[:, 1] = torch.arange(1, 4)
        self.class_tokens[:, 2] = 31

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        return self.images[index], self.class_tokens[label], label


class DropoutTests(unittest.TestCase):
    def test_zero_dropout_matches_original_outputs_and_gradients(self):
        for mask in (False, True):
            with self.subTest(mask=mask):
                old = LegacyBlock(mask)
                new = ResAttenBlock(16, 2, 32, mask, dropout=0.0)
                new.load_state_dict(old.state_dict(), strict=True)
                old_input = torch.randn(4, 8, 16, requires_grad=True)
                new_input = old_input.detach().clone().requires_grad_()
                old_output, new_output = old(old_input), new(new_input)
                torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
                old_output.square().sum().backward()
                new_output.square().sum().backward()
                torch.testing.assert_close(new_input.grad, old_input.grad, rtol=0, atol=0)
                for (old_name, old_param), (new_name, new_param) in zip(old.named_parameters(), new.named_parameters()):
                    self.assertEqual(old_name, new_name)
                    torch.testing.assert_close(new_param.grad, old_param.grad, rtol=0, atol=0)

    def test_dropout_scope_and_legacy_constructor_default(self):
        self.assertEqual(config.image_dropout, 0.1)
        model = clip(**SMALL_CONFIG, image_dropout=0.1)
        image_layers = [m for m in model.img_encoder.modules() if isinstance(m, nn.Dropout)]
        text_layers = [m for m in model.text_encoder.modules() if isinstance(m, nn.Dropout)]
        self.assertEqual(len(image_layers), 3 * SMALL_CONFIG["num_layers"])
        self.assertTrue(all(m.p == 0.1 for m in image_layers))
        self.assertTrue(text_layers)
        self.assertTrue(all(m.p == 0.0 for m in text_layers))
        # All old positional arguments still work, without enabling dropout.
        legacy_call = clip(*SMALL_CONFIG.values())
        self.assertTrue(all(m.p == 0.0 for m in legacy_call.modules() if isinstance(m, nn.Dropout)))
        for block in model.text_encoder.transformer.layes:
            self.assertIsInstance(block.mlp[0], nn.Linear)
            self.assertIsInstance(block.mlp[2], nn.Linear)

    def test_train_is_stochastic_eval_is_deterministic(self):
        model = clip(**SMALL_CONFIG, image_dropout=0.1)
        data = SyntheticCIFAR()
        images, tokens = data.images, data.class_tokens
        model.train()
        torch.manual_seed(1)
        first = model.encode_image(images)
        torch.manual_seed(2)
        second = model.encode_image(images)
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(model.encode_text(tokens), model.encode_text(tokens), rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model.encode_image(images), model.encode_image(images), rtol=0, atol=0)

    def test_old_model_and_optimizer_state_load_strictly(self):
        old_model = clip(**SMALL_CONFIG)
        old_model.img_encoder.Transformer.layes = nn.ModuleList(LegacyBlock() for _ in range(2))
        old_model.text_encoder.transformer.layes = nn.ModuleList(LegacyBlock(True) for _ in range(2))
        old_optimizer = create_optimizer(old_model, 3e-4, 0.01)
        # Populate Adam state for every parameter without running a full model.
        for parameter in old_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        old_optimizer.step()
        new_model = clip(**SMALL_CONFIG, image_dropout=0.1)
        new_model.load_state_dict(old_model.state_dict(), strict=True)
        self.assertEqual(list(dict(old_model.named_parameters())), list(dict(new_model.named_parameters())))
        new_optimizer = create_optimizer(new_model, 3e-4, 0.01)
        new_optimizer.load_state_dict(copy.deepcopy(old_optimizer.state_dict()))
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
            torch.testing.assert_close(old_optimizer.state[old_param]["exp_avg"], new_optimizer.state[new_param]["exp_avg"])
        old_model.eval()
        new_model.eval()
        images = SyntheticCIFAR().images
        torch.testing.assert_close(old_model.encode_image(images), new_model.encode_image(images), rtol=0, atol=0)

    def test_training_entrypoint_and_saved_config_inference(self):
        # Exercise the real CLI, training loop, evaluation, and checkpoint writer;
        # only replace CIFAR download and TensorBoard with synthetic fixtures.
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            snapshot = "class config:\n" + "".join(f"    {key} = {value!r}\n" for key, value in SMALL_CONFIG.items())
            config_path = temp / "config.py"
            config_path.write_text(snapshot + "    image_dropout = 0.1\n")
            argv = ["train.py", "--n", "dropout_test", "--save_dir", str(temp),
                    "--num_epochs", "1", "--warmup_epochs", "0", "--num_workers", "0",
                    "--batch_size", "3", "--precision", "fp32"]
            with ExitStack() as stack:
                stack.enter_context(patch("sys.argv", argv))
                stack.enter_context(patch("config.config", SimpleNamespace(**SMALL_CONFIG, image_dropout=0.1)))
                stack.enter_context(patch("data.dataset.Dataset", SyntheticCIFAR))
                stack.enter_context(patch("torch.utils.tensorboard.SummaryWriter"))
                stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                stack.enter_context(patch("pathlib.Path.read_bytes", autospec=True,
                                          side_effect=lambda path: config_path.read_text().encode()))
                # Close FileHandlers before TemporaryDirectory cleanup.
                import logging
                previous_handlers = logging.root.handlers[:]
                try:
                    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
                finally:
                    for handler in logging.root.handlers[:]:
                        if handler not in previous_handlers:
                            logging.root.removeHandler(handler)
                            handler.close()
            experiment = temp / "expdropout_test"
            self.assertIn("image_dropout: 0.1", (experiment / "info.txt").read_text())
            checkpoint_path = experiment / "last.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertTrue(all(torch.isfinite(value).all() for value in checkpoint["model"].values()))
            for probability in (0.1, None):
                with self.subTest(saved_probability=probability):
                    saved = snapshot + ("    image_dropout = 0.1\n" if probability is not None else "")
                    (experiment / "config.py").write_text(saved)
                    saved_config = zero_shot.load_config(checkpoint_path)
                    with patch.object(zero_shot, "config", saved_config, create=True):
                        loaded = zero_shot.load_model(checkpoint_path, torch.device("cpu"))
                    dropouts = [m for m in loaded.img_encoder.modules() if isinstance(m, nn.Dropout)]
                    self.assertTrue(all(m.p == (probability or 0.0) for m in dropouts))
                    direct = clip(**SMALL_CONFIG, image_dropout=0.1).eval()
                    direct.load_state_dict(checkpoint["model"], strict=True)
                    images = SyntheticCIFAR().images
                    torch.testing.assert_close(loaded.encode_image(images), direct.encode_image(images), rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
