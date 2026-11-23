"""Model evaluation and metrics reporting utilities."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score


def get_performance_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute overall accuracy, macro F1, weighted F1, and detailed report.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.

    Returns:
        Dictionary with accuracy, macro_f1, weighted_f1, and classification_report.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
