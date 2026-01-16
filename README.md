# Benchmark Inference for Vuln Detection

> vLLM Serve + OpenAI-Compatible SDK

本项目通过 `vllm serve` 启动 OpenAI-compatible 服务，然后使用 OpenAI SDK 进行推理。
目前支持两种Openai-compatible 接口：
- `chat`: `POST /v1/chat/completions`
- `responses`: `POST /v1/responses`
配置文件使用 `main` 指向当前要用的模型，推理时自动读取对应配置。

## 运行方式

方式一：先手动启动服务，再推理
```
./scripts/launch_vllm.sh config/models.json vllm_chat_local
./scripts/run_inference.sh config/models.json vllm_chat_local \
  prompt/simple_vuln_prompt.txt --no_launch
```

方式二：推理脚本自动启动服务
```
./scripts/run_inference.sh config/models.json vllm_chat_local \
  prompt/simple_vuln_prompt.txt
```

如果你已经手动启动了服务，请加 `--no_launch`，避免重复启动。

## 输出说明

默认会生成以下 4 类文件，并且同一次运行的文件名保持一致：
- `logs/vllm/<run_name>.log`
- `logs/infer/<run_name>.log`
- `reports/outputs/<run_name>.jsonl`
- `reports/metrics/<run_name>.csv`

`run_name` 默认格式为 `[模型名]_YYYYMMDD-HHMMSS`，也可通过 `--run_name` 或 `--output` 自定义。

## 数据输入（可选）

如果需要评估指标，请传入 `--data_file`（支持 `.jsonl` / `.json` / `.csv`）。
数据行需要包含标签字段（默认 `label`），并可用于渲染 prompt 模板中的占位符：

```json
{"id": "sample-1", "code": "int a=0;", "label": true}
```

## 配置说明（config/models.json）

关键结构：
- `main`: 默认模型 key（不传 `--model` 时使用）
- `models`: 模型配置映射表

常用字段：
- `interface`: `"chat"` 或 `"responses"`
- `model_name`: 模型名（用于 OpenAI-compatible 调用）
- `base_url`: 例如 `http://127.0.0.1:8000/v1` 或远端 API URL
- `api_key`: 远端或本地服务的 key（本地一般是 `EMPTY`）
- `generation`: 推理参数，例如 `temperature`、`max_tokens` 或 `max_output_tokens`
- `serve`: vLLM 服务启动参数（可选）
  - `enabled`: 是否允许自动启动（默认 true）
  - `host` / `port`
  - `model_path`
  - `tensor_parallel_size`
  - `max_model_len`
  - `dtype`
  - `gpu_memory_utilization`
  - `log_file`

Example:
```json
{
  "main": "vllm_chat_local",
  "models": {
    "vllm_chat_local": {
      "interface": "chat",
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key": "EMPTY",
      "rate_limit_delay": 0.0,
      "generation": {
        "temperature": 0.0,
        "max_tokens": 256
      },
      "serve": {
        "enabled": true,
        "host": "127.0.0.1",
        "port": 8000,
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "log_file": "logs/vllm_chat_local.log"
      }
    },
    "vllm_responses_local": {
      "interface": "responses",
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key": "EMPTY",
      "rate_limit_delay": 0.0,
      "generation": {
        "temperature": 0.0,
        "max_output_tokens": 256
      },
      "serve": {
        "enabled": true,
        "host": "127.0.0.1",
        "port": 8000,
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "log_file": "logs/vllm_responses_local.log"
      }
    },
    "remote_chat_api": {
      "interface": "chat",
      "model_name": "your-remote-model",
      "base_url": "https://api.example.com/v1",
      "api_key": "YOUR_API_KEY",
      "rate_limit_delay": 0.2,
      "generation": {
        "temperature": 0.2,
        "max_tokens": 512
      }
    }
  }
}

```

如果使用远端 API，不需要 `serve` 段，或者设置 `"enabled": false`。

## 依赖

```
pip install openai vllm
```

> `vllm` 只在你需要启动本地服务时必需。
