from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModel(ABC):
    """
    Unified interface for all models (local or remote).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate outputs for a batch of prompts.

        Args:
            prompts: list of input strings
        Returns:
            list of generated strings
        """
        pass
