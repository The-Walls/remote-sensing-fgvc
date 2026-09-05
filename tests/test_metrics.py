"""metrics.summarize / top_confusions on a hand-computed confusion matrix."""
import math
import unittest

import numpy as np

from src import metrics


class MetricsTest(unittest.TestCase):
    def test_summary_matches_hand_computation(self):
        # rows = true, cols = pred. class 2 never appears in val.
        cm = np.array([[2, 0, 0],
                       [1, 1, 0],
                       [0, 0, 0]])
        s = metrics.summarize(cm)
        self.assertAlmostEqual(s["overall_top1"], 3 / 4)
        self.assertAlmostEqual(s["mean_per_class_top1"], (1.0 + 0.5) / 2)   # unseen class excluded
        # class0: P=2/3 R=1 F1=0.8 ; class1: P=1 R=0.5 F1=2/3
        self.assertAlmostEqual(s["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertTrue(math.isnan(s["mean_per_class_top1_support_ge5"]))
        self.assertEqual(s["n_classes_support_ge5"], 0)
        self.assertEqual(s["per_class_support"], [2, 2, 0])

    def test_support_ge5_subset(self):
        cm = np.array([[5, 0], [1, 1]])       # class0 support 5, class1 support 2
        s = metrics.summarize(cm)
        self.assertAlmostEqual(s["mean_per_class_top1_support_ge5"], 1.0)
        self.assertEqual(s["n_classes_support_ge5"], 1)

    def test_confusion_builds_from_labels(self):
        cm = metrics.confusion(np.array([0, 1, 1]), np.array([0, 0, 1]), 3)
        self.assertEqual(cm.tolist(), [[1, 0, 0], [1, 1, 0], [0, 0, 0]])

    def test_top_confusions_orders_by_count_and_skips_diagonal(self):
        cm = np.array([[9, 1, 0], [3, 5, 0], [0, 0, 4]])
        top = metrics.top_confusions(cm, ["a", "b", "c"], k=5)
        self.assertEqual([(t["true"], t["pred"], t["count"]) for t in top],
                         [("b", "a", 3), ("a", "b", 1)])
        self.assertEqual(top[0]["true_support"], 8)


if __name__ == "__main__":
    unittest.main()
