"""Train/eval transforms.

Domain difficulty #2 (arbitrary target orientation): overhead imagery has no
canonical "up", so the dihedral group + free rotation is a legitimate label-
preserving augmentation here, unlike in natural-image classification.

  basic    resize + horizontal flip
  rs_rot   resize larger, dihedral group + free rotation, centre-crop (L2)
  rs_zoom  resize larger, horizontal flip, centre-crop -- rs_rot with the
           orientation part removed, so L2 - L2z isolates orientation and
           L2z - L1 isolates the zoom/crop that rs_rot needs anyway
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

    aug = cfg["data"]["aug"]
    if aug == "basic":
        return T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
        ] + norm)

    big = math.ceil(size * ROT_PAD)   # ceil: never a sub-pixel short of the inscribed square
    if aug == "rs_rot":
        return T.Compose([
            T.Resize((big, big), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            T.RandomRotation(ROT_DEG, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
        ] + norm)

    if aug == "rs_zoom":
        return T.Compose([
            T.Resize((big, big), antialias=True),
            T.RandomHorizontalFlip(),
            T.CenterCrop(size),
        ] + norm)

    raise ValueError(f"unknown aug: {aug}")
