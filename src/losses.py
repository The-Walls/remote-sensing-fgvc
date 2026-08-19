"""Losses. Domain difficulty #4 (long tail).

cb_ce: class-balanced cross-entropy (Cui et al., CVPR 2019). Each class is
weighted by the inverse "effective number" (1 - beta^n) / (1 - beta), which
interpolates between no reweighting (beta -> 0) and inverse-frequency
(beta -> 1). beta is fixed at 0.9999 by decision and recorded in the config.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleWeightedSmoothedCE(nn.Module):
    """Label-smoothed CE with class weights applied to the target sample only.

    Passing both ``weight`` and ``label_smoothing`` to CrossEntropyLoss weights
    every class in the smoothed target. With extreme long-tail weights, that
    makes every image receive a large rare-class penalty. Here smoothing is
    computed first, then the target class selects the sample weight.
    """

    def __init__(self, weight, label_smoothing):
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        per_sample = F.cross_entropy(
            logits, target, reduction="none", label_smoothing=self.label_smoothing)
        sample_weight = self.weight[target]
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def build(cfg, train_class_counts):
    ls = cfg["loss"]["label_smoothing"]
    if cfg["loss"]["name"] == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls), None

    if cfg["loss"]["name"] == "cb_ce":
        beta = cfg["loss"]["cb_beta"]
        n = np.asarray(train_class_counts, dtype=np.float64)
        eff = (1.0 - np.power(beta, np.maximum(n, 1))) / (1.0 - beta)
        w = 1.0 / eff
        w = w / w.sum() * len(w)            # mean weight 1 -> loss scale unchanged
        t = torch.as_tensor(w, dtype=torch.float32)
        return SampleWeightedSmoothedCE(t, ls), w.tolist()

    raise ValueError(f"unknown loss: {cfg['loss']['name']}")
