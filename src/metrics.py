"""Metrics. On a 274x-imbalanced dataset overall top-1 is dominated by the head
classes, so mean-per-class top-1 and macro-F1 are reported alongside it.
"""
import numpy as np


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def summarize(cm):
    support = cm.sum(1)
    correct = np.diag(cm)
    seen = support > 0

    per_class = np.zeros(len(support), dtype=np.float64)
    per_class[seen] = correct[seen] / support[seen]

    pred_tot = cm.sum(0)
    prec = np.divide(correct, pred_tot, out=np.zeros_like(correct, float),
                     where=pred_tot > 0)
    rec = per_class
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)

    return {
        "overall_top1": float(correct.sum() / max(cm.sum(), 1)),
        "mean_per_class_top1": float(per_class[seen].mean()),
        "macro_f1": float(f1[seen].mean()),
        # tail classes can hold a single val image; a mean over those is noise,
        # so the well-supported subset is reported next to the full mean
        "mean_per_class_top1_support_ge5": float(
            per_class[support >= 5].mean()) if (support >= 5).any() else float("nan"),
        "n_classes_support_ge5": int((support >= 5).sum()),
        "per_class_top1": per_class.tolist(),
        "per_class_support": support.tolist(),
    }


def top_confusions(cm, classes, k=15):
    """Most frequent (true, pred) off-diagonal pairs."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0]
    out = []
    for i, j in idx[:k]:
        if off[i, j] == 0:
            break
        out.append({"true": classes[i], "pred": classes[j], "count": int(off[i, j]),
                    "true_support": int(cm[i].sum())})
    return out
