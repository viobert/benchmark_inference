"""
Run inference for vulnerability detection benchmark.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from src.utils.metrics import compute_binary_metrics
from src.utils.output_parser import extract_prediction
from src.models.registry import build_model
from src.server.vllm_server import ensure_vllm_server

# ---------------------------------------------------------------------------
# Logger (module-level, no passing around)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Avoid duplicate handlers, but allow switching log files across runs.
    existing_files = {
        getattr(h, "baseFilename", None) for h in root.handlers
        if isinstance(h, logging.FileHandler)
    }
    log_file = str(log_path.resolve())
    if log_file not in existing_files:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        root.addHandler(sh)
    return root

# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

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
    cfg["model_name"] = key
    cfg.setdefault("interface", "chat")
    if "base_url" not in cfg:
        serve_cfg = cfg.get("serve")
        if serve_cfg and serve_cfg.get("enabled", True):
            cfg["base_url"] = build_base_url_from_serve(serve_cfg)
        else:
            raise ValueError(
                "Missing required field: base_url (needed for remote API models)."
            )
    return cfg, key


def build_default_run_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._\-\[\]]+", "-", model_name)
    cleaned = cleaned.strip("._-") or "run"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"[{cleaned}]_{timestamp}"


def build_base_url_from_serve(serve_cfg: dict) -> str:
    host = serve_cfg.get("host") or "127.0.0.1"
    port = serve_cfg.get("port") or 8000
    return f"http://{host}:{port}/v1"

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def col_fn(
    prompt_file: Path,
    dataset_path: Path,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load a HuggingFace Dataset and add prompt.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    try:
        dataset = load_from_disk(str(dataset_path))
        if isinstance(dataset, DatasetDict) or isinstance(dataset, dict):
            if not dataset:
                raise ValueError("Dataset dict is empty.")
            first_split = next(iter(dataset.keys()))
            logger.warning(
                "Dataset is a dict; using first split '%s'.", first_split
            )
            dataset = dataset[first_split]
    except Exception:
        logger.exception("Failed to load dataset from %s", dataset_path)
        raise

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
        logger.info("Limiting to first %d samples", limit)

    template = prompt_file.read_text(encoding="utf-8")

    def _add_prompt(record: dict) -> dict:
        prompt_record = {
            k: v for k, v in record.items() if k not in ("label", "labels")
        }
        try:
            prompt = template.format_map(prompt_record)
        except KeyError as e:
            raise KeyError(
                f"Missing field {e} required by prompt template"
            ) from e

        return {
            "prompt": prompt,
        }

    logger.info("Building prompts for %d samples", len(dataset))
    return dataset.map(_add_prompt)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no_launch", action="store_true")
    parser.add_argument("--launch_timeout", type=float, default=120.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N samples from the dataset.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    config_path = Path(args.config)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    
    prompt_file_value = args.prompt_file or config_data.get("prompt_file")
    dataset_value = args.dataset or config_data.get("dataset")
    if not prompt_file_value or not dataset_value:
        raise ValueError(
            "Missing prompt_file or dataset; pass flags or set them in config."
        )

    prompt_file = Path(prompt_file_value)
    dataset_path = Path(dataset_value)

    model_cfg, model_key = load_model_config(config_path, args.model)
    model_name = model_cfg.get("model_name", model_key)

    base_name = build_default_run_name(model_name)

    vllm_log_path = Path("logs/vllm") / f"{base_name}.log"
    infer_log_path = Path("logs/infer") / f"{base_name}.log"
    output_path = Path("reports") / f"{base_name}.jsonl"

    setup_logger(infer_log_path)

    logger.info("Run name: %s", base_name)
    logger.info("Model: %s", model_key)
    logger.info("Base URL: %s", model_cfg.get("base_url", ""))
    logger.info("Dataset: %s", dataset_path)

    if not args.no_launch:
        serve_cfg = model_cfg.get("serve")
        if serve_cfg and serve_cfg.get("enabled", True):
            serve_cfg.setdefault("log_file", str(vllm_log_path))

        ensure_vllm_server(
            model_cfg,
            startup_timeout=args.launch_timeout,
        )

    samples = col_fn(prompt_file, dataset_path, limit=args.limit)
    if len(samples) == 0:
        logger.error("Dataset is empty after prompt construction.")
        raise SystemExit(1)

    model = build_model(model_cfg)

    results: List[dict] = []
    predictions: List[Optional[bool]] = []

    tp = tn = fp = fn = 0

    progress = tqdm(range(len(samples)), desc="Infer", unit="sample")

    for idx in progress:
        sample = samples[idx]

        outputs = model.generate(prompts=[sample["prompt"]])
        output = outputs[0] if outputs else ""

        pred = extract_prediction(output)
        id = sample['id']
        label = sample["vul"]

        predictions.append(pred)

        if label is not None and pred is not None:
            if label and pred:
                tp += 1
            elif not label and not pred:
                tn += 1
            elif not label and pred:
                fp += 1
            elif label and not pred:
                fn += 1

        progress.set_postfix(tp=tp, tn=tn, fp=fp, fn=fn)

        results.append(
            {
                "id": str(id),
                "output": output,
                "label": label,
                "prediction": pred,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_binary_metrics(
        labels=[s["vul"] for s in samples],
        preds=predictions,
    )

    logger.info(
        "Metrics: tp=%s tn=%s fp=%s fn=%s acc=%.6f prec=%.6f recall=%.6f f1=%.6f",
        metrics.get("tp", 0),
        metrics.get("tn", 0),
        metrics.get("fp", 0),
        metrics.get("fn", 0),
        metrics.get("acc", 0.0),
        metrics.get("prec", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
    )

    logger.info("Saved %d results to %s", len(results), output_path)


if __name__ == "__main__":
    main()
