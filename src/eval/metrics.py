import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

def all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob))
    }
