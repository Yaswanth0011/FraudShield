"""
SentinelShield AI - Evaluation Metrics & Performance Benchmarks
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def evaluate_model_performance(model, X_test, y_test) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics on test data.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def generate_evaluation_report(model, X_test, y_test) -> str:
    """Generate textual classification report."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"])
