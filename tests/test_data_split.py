"""split_indices: a duplicate group must never straddle train/val."""
import unittest

from src.data import split_indices


def _toy():
    # 3 classes. class 0: 4 groups of 2 twins. class 1: 5 singletons.
    # class 2: one group of 3 that cannot be split without leaking.
    labels, groups = [], []
    for g in range(4):
        labels += [0, 0]
        groups += [f"a{g}", f"a{g}"]
    for g in range(5):
        labels.append(1)
        groups.append(f"b{g}")
    labels += [2, 2, 2]
    groups += ["c0", "c0", "c0"]
    return labels, groups


class SplitTest(unittest.TestCase):
    def test_partition(self):
        labels, groups = _toy()
        for group_aware in (True, False):
            tr, va = split_indices(labels, groups, 0.25, 0, 1, group_aware)
            self.assertEqual(sorted(tr + va), list(range(len(labels))))
            self.assertFalse(set(tr) & set(va))

    def test_group_aware_keeps_groups_atomic(self):
        labels, groups = _toy()
        for seed in range(5):
            tr, va = split_indices(labels, groups, 0.25, seed, 1, True)
            side = {}
            for i in tr:
                side.setdefault(groups[i], set()).add("train")
            for i in va:
                side.setdefault(groups[i], set()).add("val")
            straddling = [g for g, s in side.items() if len(s) > 1]
            self.assertEqual(straddling, [], f"seed {seed}: groups in both splits")

    def test_every_class_keeps_at_least_one_train_group(self):
        labels, groups = _toy()
        tr, va = split_indices(labels, groups, 0.25, 0, 1, True)
        self.assertEqual({labels[i] for i in tr}, {0, 1, 2})
        # class 2 is one atomic group: it cannot go to val without emptying train
        self.assertNotIn(2, {labels[i] for i in va})

    def test_min_val_per_class_respected_when_possible(self):
        labels, groups = _toy()
        tr, va = split_indices(labels, groups, 0.25, 0, 1, True)
        self.assertTrue({0, 1} <= {labels[i] for i in va})

    def test_seed_determinism(self):
        labels, groups = _toy()
        a = split_indices(labels, groups, 0.25, 0, 1, True)
        b = split_indices(labels, groups, 0.25, 0, 1, True)
        c = split_indices(labels, groups, 0.25, 1, 1, True)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
