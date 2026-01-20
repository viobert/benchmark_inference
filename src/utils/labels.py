from typing import Any, Optional

TRUE_TOKENS = {
    "true",
    "1",
    "yes",
    "y",
    "vulnerable",
    "positive",
}

FALSE_TOKENS = {
    "false",
    "0",
    "no",
    "n",
    "safe",
    "benign",
    "negative",
}


def normalize_label(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUE_TOKENS:
            return True
        if token in FALSE_TOKENS:
            return False
    return None


def label_from_id(record_id: Any) -> Optional[bool]:
    if record_id is None:
        return None
    prefix = str(record_id).split("-", 1)[0]
    if prefix == "bug":
        return True
    if prefix in ("good", "fix"):
        return False
    return None
