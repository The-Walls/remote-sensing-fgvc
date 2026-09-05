"""timm backbone + swappable pooling head.

Heads address domain difficulty #3 (small discriminative parts diluted by
background clutter under global average pooling):
  gap              -- baseline global average pooling
  compact_bilinear -- second-order feature encoding via Tensor Sketch, the
                      standard cheap stand-in for full bilinear pooling
  cbam             -- channel + spatial attention before pooling, i.e. weakly
                      supervised localisation of the discriminative region
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactBilinearPooling(nn.Module):
    """Tensor Sketch (Count Sketch + FFT) approximation of the bilinear product.

    Full bilinear pooling gives C^2 dimensions (4M for C=2048); Tensor Sketch
    projects to `out_dim` while approximately preserving inner products.
    Gao et al., "Compact Bilinear Pooling", CVPR 2016.
    """

    def __init__(self, in_dim, out_dim=8192, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        for k in (1, 2):
            self.register_buffer(f"h{k}", torch.as_tensor(
                rng.integers(0, out_dim, in_dim), dtype=torch.long))
            self.register_buffer(f"s{k}", torch.as_tensor(
                rng.integers(0, 2, in_dim) * 2 - 1, dtype=torch.float32))
        self.out_dim = out_dim

    def _sketch(self, x, h, s):
        # x: (N, HW, C) -> (N, HW, out_dim)
        out = x.new_zeros(x.shape[0], x.shape[1], self.out_dim)
        return out.index_add_(2, h, x * s)

    def forward(self, feat):                      # feat: (N, C, H, W)
        n, c, hh, ww = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(n, hh * ww, c).float()
        s1 = torch.fft.rfft(self._sketch(x, self.h1, self.s1), dim=2)
        s2 = torch.fft.rfft(self._sketch(x, self.h2, self.s2), dim=2)
        y = torch.fft.irfft(s1 * s2, n=self.out_dim, dim=2).mean(dim=1)
        y = torch.sign(y) * torch.sqrt(torch.abs(y) + 1e-8)   # signed sqrt
        return F.normalize(y, dim=1)                          # L2


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, ch, reduction=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 8)), nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 8), ch))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        n, c, _, _ = x.shape
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) +
                           self.mlp(x.amax((2, 3)))).view(n, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)))
        return x * sa


class Net(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["model"]["name"], pretrained=cfg["model"]["pretrained"],
            num_classes=0, global_pool="", drop_rate=cfg["model"]["drop_rate"],
            img_size=cfg["data"]["img_size"] if _needs_img_size(cfg["model"]["name"]) else None,
        )
        ch = self.backbone.num_features
        self.head_kind = cfg["model"]["head"]
        self.attn = CBAM(ch) if self.head_kind == "cbam" else None
        if self.head_kind == "compact_bilinear":
            self.cbp = CompactBilinearPooling(ch)
            self.fc = nn.Linear(self.cbp.out_dim, num_classes)
        else:
            self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim == 3:                       # ViT: (N, tokens, C) -> (N, C, h, w)
            n, t, c = f.shape
            s = int(round(t ** 0.5))
            f = f[:, t - s * s:, :].transpose(1, 2).reshape(n, c, s, s)
        if self.attn is not None:
            f = self.attn(f)
        if self.head_kind == "compact_bilinear":
            return self.fc(self.cbp(f))
        return self.fc(f.mean((2, 3)))


def _needs_img_size(name):
    return name.startswith(("vit_", "deit_", "swin_", "beit_"))


def build(cfg, num_classes):
    model = Net(cfg, num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def param_groups(model, base_lr, head_mult, weight_decay):
    """Backbone at base_lr, freshly-initialised head at head_mult x base_lr.

    Norm weights, biases (every 1-d parameter) and whatever the timm model lists
    in `no_weight_decay()` (ViT pos_embed / cls_token) get no weight decay, as
    in the timm ConvNeXt/ViT recipes; pulling scale parameters toward zero is
    not regularisation.
    """
    head_names = ("fc.", "attn.", "cbp.")
    skip = {f"backbone.{n}" for n in
            getattr(model.backbone, "no_weight_decay", lambda: ())()}
    buckets = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith(head_names)
        no_decay = p.ndim <= 1 or n in skip
        buckets.setdefault((is_head, no_decay), []).append(p)
    groups = []
    for (is_head, no_decay), params in sorted(buckets.items()):
        groups.append({
            "params": params,
            "lr": base_lr * (head_mult if is_head else 1.0),
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": ("head" if is_head else "backbone") + ("_no_decay" if no_decay else ""),
        })
    return groups
