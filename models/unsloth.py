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
class UnslothConfig:
    provider: str
    model_name: str = "unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit"
    display_name: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 128
    request_timeout: int = 0
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    device_map: str = "auto"
    system_prompt: Optional[str] = None
    prompt_version: str = "default"
    retry_attempts: int = 1
    retry_sleep: float = 1.0
    min_seconds_between_calls: float = 0.0
    trust_remote_code: bool = True


class UnslothLLM:
    """Local Unsloth-backed chat LLM.

    This wrapper follows the same interface as the API-backed wrappers: the
    experiment code sets ``system_prompt`` from the dataset wrapper and calls
    ``generate(prompt)``. The model is loaded once locally with Unsloth and then
    used through the tokenizer chat template.
    """

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import torch
            from unsloth import FastLanguageModel
        except Exception as exc:
            raise ImportError(
                "Install Unsloth first. See requirements.txt; in Colab this usually means "
                "`pip install unsloth` and a recent transformers version."
            ) from exc

        self.torch = torch
        self.FastLanguageModel = FastLanguageModel
        self.cfg = UnslothConfig(**cfg)
        self._last_call_time = 0.0
        self.system_prompt = (
            self.cfg.system_prompt
            if self.cfg.system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT
        )

        dtype = self._resolve_dtype(self.cfg.dtype)
        load_kwargs: Dict[str, Any] = {
            "model_name": self.cfg.model_name,
            "max_seq_length": int(self.cfg.max_seq_length),
            "dtype": dtype,
            "load_in_4bit": bool(self.cfg.load_in_4bit),
        }
        # Some Unsloth versions accept device_map / trust_remote_code and some
        # ignore or reject them. Try the explicit local-loading options first,
        # then fall back to the stable core arguments.
        if self.cfg.device_map:
            load_kwargs["device_map"] = self.cfg.device_map
        if self.cfg.trust_remote_code is not None:
            load_kwargs["trust_remote_code"] = bool(self.cfg.trust_remote_code)
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        except TypeError:
            load_kwargs.pop("device_map", None)
            load_kwargs.pop("trust_remote_code", None)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        FastLanguageModel.for_inference(self.model)

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _resolve_dtype(self, dtype_name: Optional[str]):
        if dtype_name is None or str(dtype_name).lower() in {"", "none", "null", "auto"}:
            return None
        name = str(dtype_name).lower()
        if name in {"float16", "fp16", "torch.float16"}:
            return self.torch.float16
        if name in {"bfloat16", "bf16", "torch.bfloat16"}:
            return self.torch.bfloat16
        if name in {"float32", "fp32", "torch.float32"}:
            return self.torch.float32
        raise ValueError(f"Unsupported Unsloth dtype: {dtype_name}")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_time
        wait = float(self.cfg.min_seconds_between_calls) - elapsed
        if wait > 0:
            time.sleep(wait)

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cuda" if self.torch.cuda.is_available() else "cpu"

    def _apply_chat_template(self, prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except TypeError:
            # Fallback for older tokenizers that do not expose all keyword args.
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(text, return_tensors="pt").input_ids

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, int(self.cfg.retry_attempts) + 1):
            try:
                self._throttle()
                input_ids = self._apply_chat_template(prompt).to(self._device())
                input_len = int(input_ids.shape[-1])

                generation_kwargs: Dict[str, Any] = {
                    "input_ids": input_ids,
                    "max_new_tokens": int(self.cfg.max_output_tokens),
                    "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                    "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                }
                if float(self.cfg.temperature) <= 0:
                    generation_kwargs["do_sample"] = False
                else:
                    generation_kwargs["do_sample"] = True
                    generation_kwargs["temperature"] = float(self.cfg.temperature)

                with self.torch.inference_mode():
                    output_ids = self.model.generate(**generation_kwargs)

                self._last_call_time = time.time()
                new_tokens = output_ids[0, input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return str(text if text is not None else "").strip()

            except Exception as exc:
                last_err = exc
                if attempt >= int(self.cfg.retry_attempts):
                    break
                wait = float(self.cfg.retry_sleep) * attempt
                print(
                    f"[main_llm:unsloth] attempt {attempt} failed: {exc}. "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"Unsloth local generation failed after retries: {last_err}")
