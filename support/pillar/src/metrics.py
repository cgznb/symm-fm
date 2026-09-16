import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score)

METRIC_KEYS = ("acc", "auroc", "sens", "spec", "prec", "npv", "bacc", "prauc")


def compute_metrics(y_true, y_prob, threshold=0.5):
    """The 8 reported metrics at `threshold`: Acc, AUROC, Sens, Spec, Prec, NPV, BACC, PRAUC.
    Sens=recall/TPR, Spec=TNR, Prec=PPV, NPV=TN/(TN+FN). Ill-defined ratios -> nan."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def ratio(num, den):
        return float(num) / den if den else float("nan")

    return {
        "acc": ratio(tp + tn, tp + tn + fp + fn),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "sens": ratio(tp, tp + fn),
        "spec": ratio(tn, tn + fp),
        "prec": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "prauc": float(average_precision_score(y_true, y_prob)),
    }
