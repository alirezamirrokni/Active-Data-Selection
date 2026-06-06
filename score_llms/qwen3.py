from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

import numpy as np


class Qwen3ScoreLLM:
    """Frozen Qwen3-family feature model used only for the online score \\tilde{eta}."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError(
                "Install torch, transformers, accelerate, and bitsandbytes first. "
                "In Colab, run: "
                "`pip install -U transformers accelerate bitsandbytes`."
            ) from exc

        self.torch = torch
        self.cfg = dict(cfg)

        self.model_name = self.cfg.get("model_name", "Qwen/Qwen3-8B")
        self.max_length = int(self.cfg.get("max_length", 1024))
        self.encode_batch_size = int(self.cfg.get("encode_batch_size", 4))
        self.pooling = str(self.cfg.get("pooling", "mean")).lower()
        self.normalize_features = bool(self.cfg.get("normalize_features", True))
        self.cache_dir = self.cfg.get("cache_dir", None)

        self.device_map = self.cfg.get("device_map", "auto")
        self.torch_dtype = self.cfg.get("torch_dtype", "auto")
        self.load_in_4bit = bool(self.cfg.get("load_in_4bit", True))
        self.load_in_8bit = bool(self.cfg.get("load_in_8bit", False))

        self.prompt_template = self.cfg.get(
            "prompt_template",
            "Question:\n{question}\n\nModel answer:\n{model_answer}\n",
        )

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Use only one of load_in_4bit=True or load_in_8bit=True, not both.")

        if self.pooling not in {"mean", "last"}:
            raise ValueError(f"Unknown pooling='{self.pooling}'. Use 'mean' or 'last'.")

        print(f"[score_llm] loading frozen feature model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        dtype = self._resolve_torch_dtype(self.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(
                    self.cfg.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=self._resolve_torch_dtype(
                    self.cfg.get("bnb_4bit_compute_dtype", "float16")
                )
                or torch.float16,
            )

            # Avoid passing torch_dtype alongside quantization_config in some
            # transformer/bitsandbytes combinations where dtype inference is cleaner.
            if self.cfg.get("omit_torch_dtype_when_quantized", True):
                model_kwargs.pop("torch_dtype", None)

        elif self.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            if self.cfg.get("omit_torch_dtype_when_quantized", True):
                model_kwargs.pop("torch_dtype", None)

        try:
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "quantization_config" in msg or "BitsAndBytesConfig" in msg:
                raise RuntimeError(
                    "The installed transformers/bitsandbytes versions do not support "
                    "this quantization path cleanly. Upgrade with:\n"
                    "pip install -U transformers accelerate bitsandbytes"
                ) from exc
            raise

        self.model.eval()

        try:
            self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        except Exception:
            self.hidden_size = None

        print(
            "[score_llm] loaded "
            f"model={self.model_name} "
            f"pooling={self.pooling} "
            f"max_length={self.max_length} "
            f"batch_size={self.encode_batch_size} "
            f"4bit={self.load_in_4bit} "
            f"8bit={self.load_in_8bit}"
        )

    def _resolve_torch_dtype(self, value: Any):
        torch = self.torch

        if value is None:
            return None

        if value == "auto":
            return None

        if not isinstance(value, str):
            return value

        value = value.lower()

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }

        if value not in mapping:
            raise ValueError(
                f"Unknown torch dtype '{value}'. "
                "Use one of: auto, float16, fp16, bfloat16, bf16, float32, fp32."
            )

        return mapping[value]

    def format_text(self, question: str, model_answer: str) -> str:
        return self.prompt_template.format(
            question=str(question),
            model_answer=str(model_answer),
        )

    def _input_device(self):
        """Return a safe device for input tensors.

        With device_map='auto', modules may be sharded. For a single-GPU Colab
        run, sending inputs to cuda:0 is usually correct. If no CUDA is
        available, use CPU.
        """
        torch = self.torch

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def encode_rows(self, rows: List[dict]) -> np.ndarray:
        texts = [
            self.format_text(
                question=row.get("question", ""),
                model_answer=row.get("model_answer", ""),
            )
            for row in rows
        ]
        return self.encode_texts(texts)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            dim = self.hidden_size if self.hidden_size is not None else 0
            return np.empty((0, dim), dtype=np.float32)

        features = []
        batch_size = max(1, int(self.encode_batch_size))

        start = 0
        while start < len(texts):
            batch = texts[start : start + batch_size]

            try:
                batch_features = self._encode_batch(batch)
                features.append(batch_features)
                start += batch_size

            except RuntimeError as exc:
                message = str(exc).lower()

                if "out of memory" in message and batch_size > 1:
                    print(
                        f"[score_llm] CUDA OOM with encode_batch_size={batch_size}; "
                        f"retrying with batch_size={max(1, batch_size // 2)}"
                    )
                    self._clear_memory()
                    batch_size = max(1, batch_size // 2)
                    self.encode_batch_size = batch_size
                    continue

                raise

        return np.concatenate(features, axis=0)

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        torch = self.torch
        device = self._input_device()

        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            if getattr(out, "hidden_states", None) is not None:
                hidden = out.hidden_states[-1]
            elif getattr(out, "last_hidden_state", None) is not None:
                hidden = out.last_hidden_state
            else:
                raise RuntimeError(
                    "Model output does not contain hidden_states or last_hidden_state."
                )

            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

            if self.pooling == "mean":
                denom = mask.sum(dim=1).clamp(min=1)
                feats = (hidden * mask).sum(dim=1) / denom

            elif self.pooling == "last":
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                feats = hidden[batch_idx, lengths]

            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            feats = feats.detach().float().cpu().numpy().astype(np.float32)

        if self.normalize_features:
            denom = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
            feats = feats / denom

        return feats.astype(np.float32)

    def _clear_memory(self) -> None:
        gc.collect()
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass