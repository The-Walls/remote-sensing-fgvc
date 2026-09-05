"""cb_ce: effective-number weights, mean-1 normalisation, and the smoothing trap."""
import unittest

import torch
import torch.nn.functional as F

from src import config, losses


def _cfg(name):
    cfg = config.load("configs/base.yaml")
    cfg["loss"]["name"] = name
    return cfg


class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(8, 4)
        self.target = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])

    def test_ce_returns_plain_smoothed_ce(self):
        crit, w = losses.build(_cfg("ce"), [10, 10, 10, 10])
        self.assertIsNone(w)
        ref = F.cross_entropy(self.logits, self.target, label_smoothing=0.1)
        self.assertTrue(torch.allclose(crit(self.logits, self.target), ref))

    def test_cb_ce_with_balanced_counts_equals_plain_ce(self):
        crit, w = losses.build(_cfg("cb_ce"), [10, 10, 10, 10])
        self.assertTrue(all(abs(x - 1.0) < 1e-9 for x in w))
        ref = F.cross_entropy(self.logits, self.target, label_smoothing=0.1)
        self.assertTrue(torch.allclose(crit(self.logits, self.target), ref, atol=1e-6))

    def test_cb_ce_weights_favour_rare_classes_and_average_to_one(self):
        counts = [1000, 100, 10, 1]
        _, w = losses.build(_cfg("cb_ce"), counts)
        self.assertAlmostEqual(sum(w) / len(w), 1.0)
        self.assertEqual(w, sorted(w))                     # rarer -> larger weight
        self.assertGreater(w[-1] / w[0], 100)              # beta=0.9999 ~ inverse-frequency

    def test_weight_applies_to_target_class_only(self):
        # A huge weight on class 3 must not change the loss of samples whose
        # target is class 0 -- the trap the custom loss exists to avoid.
        w = torch.tensor([1.0, 1.0, 1.0, 1000.0])
        crit = losses.SampleWeightedSmoothedCE(w, 0.1)
        only0 = torch.zeros(4, dtype=torch.long)
        ref = F.cross_entropy(self.logits[:4], only0, label_smoothing=0.1)
        self.assertTrue(torch.allclose(crit(self.logits[:4], only0), ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
