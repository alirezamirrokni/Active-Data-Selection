import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_SYSTEM_PROMPT = """You are a careful GSM8K math solver.

Hard requirements:
- Solve the problem accurately.
- End with exactly one final line in this format:
#### <number>

The final line must contain only the marker #### followed by the numeric answer."""


@dataclass
class GeminiConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    max_output_tokens: int = 1024
    request_timeout: int = 120
    retry_attempts: int = 8
    retry_sleep: float = 2.0
    min_seconds_between_calls: float = 15.0
    system_prompt: Optional[str] = None
    prompt_version: str = "gsm8k_answer_format_v3"


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
        self._last_call_time = 0.0
        self.system_prompt = (
            self.cfg.system_prompt
            if self.cfg.system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT
        )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_time
        wait = self.cfg.min_seconds_between_calls - elapsed
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_after_seconds(exc: Exception, default: float) -> float:
        text = str(exc)
        patterns = [
            r"'retryDelay':\s*'([0-9.]+)s'",
            r'"retryDelay":\s*"([0-9.]+)s"',
            r"Please retry in\s*([0-9.]+)s",
            r"retry in\s*([0-9.]+)s",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) + 2.0
                except Exception:
                    pass
        return default

    def generate(self, prompt: str) -> str:
        last_err = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                response = self.client.models.generate_content(
                    model=self.cfg.model_name,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.cfg.temperature,
                        max_output_tokens=self.cfg.max_output_tokens,
                        system_instruction=self.system_prompt,
                    ),
                )
                self._last_call_time = time.time()
                text = getattr(response, "text", None)
                return str(text if text is not None else response).strip()
            except Exception as exc:
                last_err = exc
                base_wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    wait = self._retry_after_seconds(exc, default=max(60.0, base_wait))
                else:
                    wait = base_wait
                print(f"[main_llm:gemini] attempt {attempt} failed: {exc}. retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")
