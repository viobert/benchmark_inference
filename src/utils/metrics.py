from typing import Dict, Iterable, Optional, Union


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Union[int, float]]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "prec": prec,
        "recall": recall,
        "f1": f1,
    }
