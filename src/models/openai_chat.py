import time
from typing import Any, Dict, List, Optional

from .base import BaseModel
from src.utils.prompt_parser import parse_prompt


class ChatCompletionsModel(BaseModel):
    """
    OpenAI-compatible chat completions client.
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
            params["max_tokens"] = max_tokens
        params.update(extra_params)
        return params

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
            messages = [{"role": "user", "content": parsed["user"]}]
            if parsed["system"]:
                messages.insert(0, {"role": "system", "content": parsed["system"]})
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params,
            )
            results.append(resp.choices[0].message.content)
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)

        return results
