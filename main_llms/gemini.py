import os
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GeminiConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    max_output_tokens: int = 512
    request_timeout: int = 120
    retry_attempts: int = 4
    retry_sleep: float = 2.0


class GeminiLLM:
    """Main LLM under evaluation, accessed through the Gemini API."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise ImportError("Install google-genai from requirements.txt first.") from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or export it.")

        self.cfg = GeminiConfig(**cfg)
        self.types = types
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        last_err = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.cfg.model_name,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.cfg.temperature,
                        max_output_tokens=self.cfg.max_output_tokens,
                    ),
                )
                text = getattr(response, "text", None)
                return str(text if text is not None else response).strip()
            except Exception as exc:
                last_err = exc
                wait = self.cfg.retry_sleep * attempt
                print(f"[main_llm:gemini] attempt {attempt} failed: {exc}. retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")
