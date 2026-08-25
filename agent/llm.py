from abc import ABC, abstractmethod
import requests


class LLMBackend(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str
    ) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError


class OllamaBackend(LLMBackend):

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        if json_mode:
            payload["format"] = "json"

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]