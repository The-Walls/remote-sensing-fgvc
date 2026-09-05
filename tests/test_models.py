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

    def test_param_groups_split_head_and_exclude_norm_bias_from_decay(self):
        model, _ = models.build(self._cfg("cbam"), num_classes=7)
        groups = models.param_groups(model, 1e-4, 10.0, 0.05)
        self.assertEqual({g["name"] for g in groups},
                         {"backbone", "backbone_no_decay", "head", "head_no_decay"})
        covered = 0
        for g in groups:
            is_head = g["name"].startswith("head")
            self.assertAlmostEqual(g["lr"], 1e-3 if is_head else 1e-4)
            if g["name"].endswith("no_decay"):
                self.assertEqual(g["weight_decay"], 0.0)
                self.assertTrue(all(p.ndim <= 1 for p in g["params"]))   # norms + biases
            else:
                self.assertEqual(g["weight_decay"], 0.05)
                self.assertTrue(all(p.ndim > 1 for p in g["params"]))
            covered += sum(p.numel() for p in g["params"])
        self.assertEqual(covered, sum(p.numel() for p in model.parameters()))
        # the optimiser must accept these groups as-is (extra "name" key included)
        torch.optim.AdamW(groups)


if __name__ == "__main__":
    unittest.main()
