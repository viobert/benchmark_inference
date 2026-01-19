import time
from typing import Any, Dict, List, Optional

from .base import BaseModel
from src.utils.prompt_parser import parse_prompt


class ResponsesModel(BaseModel):
    """
    OpenAI-compatible responses client.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        default_params: Optional[Dict[str, Any]] = None,
        rate_limit_delay: float = 0.0,
    ):
        super().__init__(model_name)
        self.base_url = base_url
        self.api_key = api_key
        self.default_params = default_params or {}
        self.rate_limit_delay = rate_limit_delay

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAI-compatible mode. Install it "
                "with `pip install openai`."
            ) from exc

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY",
        )

    def _merge_params(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = dict(self.default_params)
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        params.update(extra_params)
        return params

    def _extract_text(self, resp: Any) -> str:
        output_text = getattr(resp, "output_text", None)
        if output_text:
            return output_text

        text_parts: List[str] = []
        output = getattr(resp, "output", None)
        if output:
            for item in output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for part in content:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if text:
                            text_parts.append(text)
        if text_parts:
            return "".join(text_parts)

        return str(resp)

    def generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        params = self._merge_params(temperature, max_tokens, kwargs)
        results: List[str] = []

        for prompt in prompts:
            parsed = parse_prompt(prompt)
            if parsed["system"]:
                input_payload: Any = [
                    {"role": "system", "content": parsed["system"]},
                    {"role": "user", "content": parsed["user"]},
                ]
            else:
                input_payload = parsed["user"]
            resp = self.client.responses.create(
                model=self.model_name,
                input=input_payload,
                **params,
            )
            results.append(self._extract_text(resp))
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)

        return results
