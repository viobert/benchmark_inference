#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/models.json}"
MODEL_KEY="${2:-}"

python - "$CONFIG_PATH" "$MODEL_KEY" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

from src.server.vllm_server import build_vllm_command


def load_model_config(config_path: Path, model_key: Optional[str]) -> dict:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Config file must include a non-empty 'models' map.")

    key = model_key or data.get("main")
    if not key:
        raise ValueError("Model key is required; set 'main' or pass a model key.")
    if key not in models:
        raise KeyError(f"Model '{key}' not found in config.")

    cfg = dict(models[key])
    cfg.setdefault("model_name", key)
    return cfg


config_path = Path(sys.argv[1])
model_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

cfg = load_model_config(config_path, model_key)
serve_cfg = cfg.get("serve")
if not serve_cfg:
    raise ValueError("Missing 'serve' config for this model.")

cmd = build_vllm_command(serve_cfg, cfg.get("model_name", ""))

env = os.environ.copy()
env.update(serve_cfg.get("env", {}))

log_file = serve_cfg.get("log_file")
if log_file:
    log_path = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, "a", encoding="utf-8")
    os.dup2(log_fp.fileno(), 1)
    os.dup2(log_fp.fileno(), 2)

print("[OK] Launching:", " ".join(shlex.quote(item) for item in cmd), flush=True)
os.execvpe(cmd[0], cmd, env)
PY
