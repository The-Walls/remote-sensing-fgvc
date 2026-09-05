"""config.load: `_base_` inheritance, deep-merge, and dotted --set overrides."""
import tempfile
import unittest
from pathlib import Path

from src import config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "base.yaml").write_text(
            "seed: 0\ndata:\n  img_size: 224\n  aug: basic\ntrain:\n  lr: 0.0001\n",
            encoding="utf-8")
        (d / "child.yaml").write_text(
            "_base_: base.yaml\nexp_name: child\ndata:\n  img_size: 448\n",
            encoding="utf-8")
        self.child = d / "child.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_child_overrides_one_field_and_inherits_the_rest(self):
        cfg = config.load(self.child)
        self.assertEqual(cfg["data"]["img_size"], 448)      # overridden
        self.assertEqual(cfg["data"]["aug"], "basic")       # inherited sibling
        self.assertEqual(cfg["train"]["lr"], 0.0001)        # inherited section
        self.assertNotIn("_base_", cfg)

    def test_set_overrides_are_yaml_typed(self):
        cfg = config.load(self.child, ["seed=3", "train.lr=0.01", "data.aug=rs_rot"])
        self.assertEqual(cfg["seed"], 3)
        self.assertIsInstance(cfg["seed"], int)
        self.assertAlmostEqual(cfg["train"]["lr"], 0.01)
        self.assertEqual(cfg["data"]["aug"], "rs_rot")

    def test_diff_reports_only_changed_leaves(self):
        a = config.load(self.child)
        b = config.load(self.child, ["data.img_size=320"])
        self.assertEqual(config.diff(a, b), {"data.img_size": (448, 320)})


if __name__ == "__main__":
    unittest.main()
