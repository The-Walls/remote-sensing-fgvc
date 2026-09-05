"""rs_rot: the pad factor must remove every black corner free rotation creates."""
import math
import unittest

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src import config, transforms as tf


def _black_pixels_after(pad, size=448, deg=30):
    """Rotate an all-white square padded by `pad`, centre-crop, count black."""
    big = math.ceil(size * pad)
    x = torch.ones(1, big, big)
    x = TF.rotate(x, deg, interpolation=TF.InterpolationMode.BILINEAR)
    x = TF.center_crop(x, [size, size])
    return int((x < 0.5).sum())


class RotPadTest(unittest.TestCase):
    def test_pad_factor_is_the_inscribed_square_bound(self):
        self.assertAlmostEqual(
            tf.ROT_PAD, math.cos(math.radians(30)) + math.sin(math.radians(30)))

    def test_no_black_corners_at_the_extreme_angle(self):
        for size in (224, 320, 448):
            for deg in (tf.ROT_DEG, -tf.ROT_DEG):
                self.assertEqual(_black_pixels_after(tf.ROT_PAD, size, deg), 0,
                                 f"black corner pixels at {size}px, {deg} deg")

    def test_old_pad_factor_did_leave_black_corners(self):
        # regression guard: the bug this constant fixes must be observable
        self.assertGreater(_black_pixels_after(1.15), 0)


class PipelineTest(unittest.TestCase):
    def _cfg(self, aug, size=64):
        cfg = config.load("configs/base.yaml")
        cfg["data"]["aug"] = aug
        cfg["data"]["img_size"] = size
        return cfg

    def test_output_shapes(self):
        img = Image.new("RGB", (100, 130), (255, 255, 255))
        for aug in ("basic", "rs_rot"):
            self.assertEqual(tuple(tf.build(self._cfg(aug), True)(img).shape), (3, 64, 64))
        self.assertEqual(tuple(tf.build(self._cfg("basic"), False)(img).shape), (3, 64, 64))

    def test_rs_rot_never_shows_fill_colour(self):
        torch.manual_seed(0)
        cfg = self._cfg("rs_rot")
        t = tf.build(cfg, True)
        mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
        std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        for _ in range(50):
            x = t(img) * std + mean          # back to [0, 1]
            self.assertEqual(int((x < 0.5).sum()), 0)

    def test_unknown_aug_rejected(self):
        with self.assertRaises(ValueError):
            tf.build(self._cfg("mixup"), True)


if __name__ == "__main__":
    unittest.main()
