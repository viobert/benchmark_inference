"""
Run inference for vulnerability detection benchmark.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from src.models.registry import build_model
from src.server.vllm_server import ensure_vllm_server


def load_prompts(prompt_file: Path) -> List[str]:
    """
    Load prompts from a text file.
    One prompt per line or separated by blank lines.
    """
    text = prompt_file.read_text(encoding="utf-8")
    prompts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return prompts


def load_model_config(
    config_file: Path,
    model_key: Optional[str],
) -> Tuple[dict, str]:
    """
    Load model config from a JSON file.
    """
    data = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Config file must include a non-empty 'models' map.")

    key = model_key or data.get("main")
    if not key:
        raise ValueError("Model key is required; set 'main' or pass --model.")
    if key not in models:
        raise KeyError(f"Model '{key}' not found in config.")

    cfg = dict(models[key])
    cfg.setdefault("model_name", key)
    cfg.setdefault("interface", "chat")
    return cfg, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--no_launch", action="store_true")
    parser.add_argument("--launch_timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)

    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompt_file)

    model_cfg, model_key = load_model_config(Path(args.config), args.model)

    if not args.no_launch:
        ensure_vllm_server(
            model_cfg,
            startup_timeout=args.launch_timeout,
        )

    model = build_model(model_cfg)

    outputs = model.generate(
        prompts=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results = []
    for i, (p, o) in enumerate(zip(prompts, outputs)):
        results.append({
            "id": i,
            "prompt": p,
            "output": o,
        })

    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Model: {model_key}")
    print(f"[OK] Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
