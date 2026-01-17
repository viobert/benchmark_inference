from typing import Dict, Iterable, Optional, Union


def _coerce_optional_bool(value: Optional[Union[bool, int]]) -> Optional[bool]:
    # Accept 0/1 or bool; return None for anything else.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def compute_binary_metrics(
    labels: Iterable[Optional[Union[bool, int]]],
    preds: Iterable[Optional[Union[bool, int]]],
) -> Dict[str, Union[int, float]]:
    tp = tn = fp = fn = 0
    valid = 0
    for label, pred in zip(labels, preds):
        label_bool = _coerce_optional_bool(label)
        pred_bool = _coerce_optional_bool(pred)
        if label_bool is None or pred_bool is None:
            continue
        valid += 1
        if label_bool is True and pred_bool is True:
            tp += 1
        elif label_bool is False and pred_bool is False:
            tn += 1
        elif label_bool is False and pred_bool is True:
            fp += 1
        elif label_bool is True and pred_bool is False:
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
