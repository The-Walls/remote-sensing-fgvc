"""split_indices: three-way, group-aware; a duplicate group never straddles splits."""
import unittest

from src.data import split_indices


def _toy():
    # class 0: 4 groups of 2 twins  -> test 1 / val 1 / train 2 groups
    # class 1: 5 singletons         -> test 1 / val 1 / train 3
    # class 2: one group of 3       -> train only (holding it out empties train)
    # class 3: 2 singletons         -> test 1 / train 1 / val 0
    labels, groups = [], []
    for g in range(4):
        labels += [0, 0]
        groups += [f"a{g}", f"a{g}"]
    for g in range(5):
        labels.append(1)
        groups.append(f"b{g}")
    labels += [2, 2, 2]
    groups += ["c0", "c0", "c0"]
    labels += [3, 3]
    groups += ["d0", "d1"]
    return labels, groups


def _split(seed=0, group_aware=True):
    labels, groups = _toy()
    return labels, groups, split_indices(labels, groups, 0.2, 0.2, seed, 1, group_aware)


class SplitTest(unittest.TestCase):
    def test_partition(self):
        for group_aware in (True, False):
            labels, _, (tr, va, te) = _split(group_aware=group_aware)
            self.assertEqual(sorted(tr + va + te), list(range(len(labels))))
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(tr) & set(te))
            self.assertFalse(set(va) & set(te))

    def test_group_aware_keeps_groups_atomic(self):
        for seed in range(5):
            _, groups, (tr, va, te) = _split(seed)
            side = {}
            for name, idx in (("train", tr), ("val", va), ("test", te)):
                for i in idx:
                    side.setdefault(groups[i], set()).add(name)
            straddling = [g for g, s in side.items() if len(s) > 1]
            self.assertEqual(straddling, [], f"seed {seed}: groups in more than one split")

    def test_every_class_keeps_a_training_group(self):
        labels, _, (tr, va, te) = _split()
        self.assertEqual({labels[i] for i in tr}, {0, 1, 2, 3})
        # class 2 is one atomic group: it can only live in train
        self.assertNotIn(2, {labels[i] for i in va + te})

    def test_two_group_class_goes_to_test_before_val(self):
        labels, _, (tr, va, te) = _split()
        self.assertIn(3, {labels[i] for i in te})
        self.assertNotIn(3, {labels[i] for i in va})

    def test_classes_with_enough_groups_reach_both_holdouts(self):
        labels, _, (tr, va, te) = _split()
        self.assertTrue({0, 1} <= {labels[i] for i in va})
        self.assertTrue({0, 1} <= {labels[i] for i in te})

    def test_ratios_on_a_large_class(self):
        labels = [0] * 100
        groups = list(range(100))
        tr, va, te = split_indices(labels, groups, 0.2, 0.2, 0, 1, True)
        self.assertEqual((len(tr), len(va), len(te)), (60, 20, 20))

    def test_seed_determinism(self):
        _, _, a = _split(0)
        _, _, b = _split(0)
        _, _, c = _split(1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
