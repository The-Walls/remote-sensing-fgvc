"""Heads and param groups on a small backbone; no pretrained download."""
import unittest

import torch

from src import config, models


class HeadModuleTest(unittest.TestCase):
    def test_cbam_keeps_shape_and_only_attenuates(self):
        torch.manual_seed(0)
        m = models.CBAM(16)
        x = torch.randn(2, 16, 5, 5)
        y = m(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool((y.abs() <= x.abs() + 1e-6).all()))   # two sigmoid gates

    def test_compact_bilinear_is_unit_norm(self):
        torch.manual_seed(0)
        m = models.CompactBilinearPooling(16, out_dim=64)
        y = m(torch.randn(2, 16, 4, 4))
        self.assertEqual(tuple(y.shape), (2, 64))
        self.assertTrue(torch.allclose(y.norm(dim=1), torch.ones(2), atol=1e-5))


class NetTest(unittest.TestCase):
    def _cfg(self, head):
        cfg = config.load("configs/base.yaml")
        cfg["model"].update({"name": "resnet18", "pretrained": False, "head": head})
        cfg["data"]["img_size"] = 32
        return cfg

    def test_every_head_produces_logits(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        for head in ("gap", "cbam", "compact_bilinear"):
            model, n_params = models.build(self._cfg(head), num_classes=7)
            model.eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(x).shape), (2, 7), head)
            self.assertGreater(n_params, 0)

    def test_param_groups_split_head_from_backbone(self):
        model, _ = models.build(self._cfg("cbam"), num_classes=7)
        back, head = models.param_groups(model, 1e-4, 10.0, 0.05)
        self.assertAlmostEqual(back["lr"], 1e-4)
        self.assertAlmostEqual(head["lr"], 1e-3)
        n_head = sum(p.numel() for p in head["params"])
        n_expected = sum(p.numel() for n, p in model.named_parameters()
                         if n.startswith(("fc.", "attn.", "cbp.")))
        self.assertEqual(n_head, n_expected)
        self.assertGreater(n_head, 0)
        self.assertEqual(sum(p.numel() for p in back["params"]) + n_head,
                         sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
