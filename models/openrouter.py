import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_SYSTEM_PROMPT = """You are a careful mathematical problem solver.

Hard requirements:
- Solve the problem accurately.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the final answer."""


@dataclass
class OpenRouterConfig:
    provider: str
    model_name: str = "google/gemma-4-31b-it"
    display_name: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 128
    request_timeout: int = 180
    retry_attempts: int = 8
    retry_sleep: float = 2.0
    min_seconds_between_calls: float = 0.5
    system_prompt: Optional[str] = None
    prompt_version: str = "default"
    base_url: str = "https://openrouter.ai/api/v1"
    response_format: Optional[str] = None
    retry_empty_content: bool = True
    site_url: Optional[str] = None
    app_name: Optional[str] = None


class OpenRouterLLM:
    """OpenRouter-backed LLM using the OpenAI-compatible chat-completions API."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError(
                "Install the OpenAI SDK first. Add `openai>=1.70.0` to requirements.txt."
            ) from exc

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it in your shell."
            )

        self.cfg = OpenRouterConfig(**cfg)
        default_headers: Dict[str, str] = {}

        site_url = self.cfg.site_url or os.environ.get("OPENROUTER_SITE_URL")
        app_name = self.cfg.app_name or os.environ.get("OPENROUTER_APP_NAME")
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.cfg.base_url,
            "timeout": self.cfg.request_timeout,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self.client = OpenAI(**client_kwargs)
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
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after) + 1.0
                except Exception:
                    pass

        text = str(exc)
        patterns = [
            r"try again in\s*([0-9.]+)\s*s",
            r"retry in\s*([0-9.]+)\s*s",
            r"retry after\s*([0-9.]+)\s*s",
            r"Retry-After[:= ]+([0-9.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) + 1.0
                except Exception:
                    pass
        return default

    def _request_kwargs(self, prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_output_tokens),
        }

        if self.cfg.response_format:
            if self.cfg.response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {"type": str(self.cfg.response_format)}

        return kwargs

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                response = self.client.chat.completions.create(**self._request_kwargs(prompt))
                self._last_call_time = time.time()
                text = str(response.choices[0].message.content or "").strip()
                if self.cfg.retry_empty_content and not text:
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    raise RuntimeError(
                        "OpenRouter returned empty message.content"
                        + (f" (finish_reason={finish_reason})" if finish_reason else "")
                    )
                return text

            except Exception as exc:
                last_err = exc
                base_wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = self._retry_after_seconds(exc, default=max(30.0, base_wait))
                else:
                    wait = base_wait
                print(
                    f"[main_llm:openrouter] attempt {attempt} failed: {exc}. "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
