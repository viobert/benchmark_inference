from typing import Dict, Iterable, Optional, Union


def compute_binary_metrics(
    labels: Iterable[Optional[bool]],
    preds: Iterable[Optional[bool]],
) -> Dict[str, Union[int, float]]:
    tp = tn = fp = fn = 0
    valid = 0
    for label, pred in zip(labels, preds):
        if label is None or pred is None:
            continue
        valid += 1
        if label is True and pred is True:
            tp += 1
        elif label is False and pred is False:
            tn += 1
        elif label is False and pred is True:
            fp += 1
        elif label is True and pred is False:
            fn += 1

    acc = (tp + tn) / valid if valid else 0.0
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
