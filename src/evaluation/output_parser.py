import json
import re
from typing import Optional

from src.utils.labels import normalize_label

JSON_KEYS = (
    "vulnerable",
    "is_vulnerable",
    "vuln",
    "label",
    "prediction",
    "pred",
)


def extract_prediction(text: str, custom_regex: Optional[str] = None) -> Optional[bool]:
    if not text:
        return None

    if custom_regex:
        match = re.search(custom_regex, text, flags=re.IGNORECASE)
        if match:
            token = match.group(1) if match.lastindex else match.group(0)
            parsed = normalize_label(token)
            if parsed is not None:
                return parsed

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for key in JSON_KEYS:
                if key in obj:
                    return normalize_label(obj[key])

    match = re.search(
        r"\bvulnerable\s*[:=\-]\s*(true|false|yes|no|1|0)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_label(match.group(1))

    return None
