import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def is_server_ready(base_url: str, timeout: float = 1.0) -> bool:
    try:
        url = _join_url(base_url, "models")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ValueError):
        return False


def build_vllm_command(
    serve_cfg: Dict[str, Any],
    model_name: str,
) -> List[str]:
    model_path = serve_cfg.get("model_path", model_name)
    cmd = ["vllm", "serve", model_path]

    host = serve_cfg.get("host")
    port = serve_cfg.get("port")
    if host:
        cmd += ["--host", str(host)]
    if port:
        cmd += ["--port", str(port)]

    tp_size = serve_cfg.get("tensor_parallel_size")
    if tp_size:
        cmd += ["--tensor-parallel-size", str(tp_size)]

    max_model_len = serve_cfg.get("max_model_len")
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]

    dtype = serve_cfg.get("dtype")
    if dtype:
        cmd += ["--dtype", str(dtype)]

    gpu_memory_utilization = serve_cfg.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

    quantization = serve_cfg.get("quantization")
    if quantization:
        cmd += ["--quantization", str(quantization)]

    served_model_name = serve_cfg.get("served_model_name")
    if served_model_name:
        cmd += ["--served-model-name", str(served_model_name)]

    if serve_cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if serve_cfg.get("enforce_eager"):
        cmd.append("--enforce-eager")

    extra_args = serve_cfg.get("extra_args")
    if extra_args:
        cmd.extend([str(arg) for arg in extra_args])

    return cmd


def ensure_vllm_server(
    model_cfg: Dict[str, Any],
    startup_timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> Optional[subprocess.Popen]:
    serve_cfg = model_cfg.get("serve")
    if not serve_cfg:
        return None
    if serve_cfg.get("enabled", True) is False:
        return None

    base_url = model_cfg.get("base_url")
    if not base_url:
        raise ValueError("base_url is required when serve is enabled.")

    if is_server_ready(base_url):
        return None

    cmd = build_vllm_command(serve_cfg, model_cfg.get("model_name", ""))

    env = os.environ.copy()
    env.update(serve_cfg.get("env", {}))

    log_file = serve_cfg.get("log_file")
    stdout = None
    stderr = None
    if log_file:
        log_path = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_fp = open(log_path, "a", encoding="utf-8")
        stdout = log_fp
        stderr = log_fp

    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        env=env,
        start_new_session=True,
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if is_server_ready(base_url):
            return proc
        time.sleep(poll_interval)

    proc.terminate()
    raise RuntimeError("vLLM server did not become ready in time.")
