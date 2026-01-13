from .openai_chat import ChatCompletionsModel
from .openai_responses import ResponsesModel


def build_model(cfg: dict):
    """
    Build model from config dict.
    """
    if "model_name" not in cfg:
        raise ValueError("Missing required field: model_name")
    if "base_url" not in cfg:
        raise ValueError("Missing required field: base_url")

    interface = cfg.get("interface", "chat")
    default_params = cfg.get("generation", {})
    rate_limit_delay = cfg.get("rate_limit_delay", 0.0)

    if interface == "chat":
        return ChatCompletionsModel(
            model_name=cfg["model_name"],
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", ""),
            default_params=default_params,
            rate_limit_delay=rate_limit_delay,
        )

    if interface == "responses":
        return ResponsesModel(
            model_name=cfg["model_name"],
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", ""),
            default_params=default_params,
            rate_limit_delay=rate_limit_delay,
        )

    raise ValueError(f"Unknown interface: {interface}")
