"""Train/eval transforms.

Domain difficulty #2 (arbitrary target orientation): overhead imagery has no
canonical "up", so the dihedral group + free rotation is a legitimate label-
preserving augmentation here, unlike in natural-image classification. `rs_rot`
is the only thing that differs from `basic`, so the L1->L2 comparison isolates it.
"""
import math

import torchvision.transforms as T

ROT_DEG = 30
# Rotate a larger image, then centre-crop away the black corners free rotation
# would otherwise introduce. The largest axis-aligned square inside a square of
# side S rotated by theta has side S/(cos+sin), so the pad factor must be at
# least cos(theta)+sin(theta) = 1.366 at 30 degrees. The previous value 1.15
# only covered +-9.4 deg and left black corners in 66.5% of training samples.
ROT_PAD = math.cos(math.radians(ROT_DEG)) + math.sin(math.radians(ROT_DEG))


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
        big = math.ceil(size * ROT_PAD)   # ceil: never a sub-pixel short of the inscribed square
        return T.Compose([
            T.Resize((big, big), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            T.RandomRotation(ROT_DEG, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
        ] + norm)

    raise ValueError(f"unknown aug: {cfg['data']['aug']}")
