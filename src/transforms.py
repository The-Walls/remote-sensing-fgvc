"""Train/eval transforms.

Domain difficulty #2 (arbitrary target orientation): overhead imagery has no
canonical "up", so the dihedral group + free rotation is a legitimate label-
preserving augmentation here, unlike in natural-image classification. `rs_rot`
is the only thing that differs from `basic`, so the L1->L2 comparison isolates it.
"""
import torchvision.transforms as T

ROT_PAD = 1.15   # rotate a slightly larger image, then centre-crop away the
                 # black corners free rotation would otherwise introduce


def build(cfg, train: bool):
    size = cfg["data"]["img_size"]
    norm = [T.ToTensor(), T.Normalize(cfg["data"]["mean"], cfg["data"]["std"])]

    if not train:
        return T.Compose([T.Resize((size, size), antialias=True)] + norm)

    if cfg["data"]["aug"] == "basic":
        return T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
        ] + norm)

    if cfg["data"]["aug"] == "rs_rot":
        big = int(round(size * ROT_PAD))
        return T.Compose([
            T.Resize((big, big), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            T.RandomRotation(30, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
        ] + norm)

    raise ValueError(f"unknown aug: {cfg['data']['aug']}")
