from typing import Any, Dict, List

import numpy as np


class Qwen3ScoreLLM:
    """Frozen Qwen3-family feature model used only for the online score \tilde{eta}."""

    def __init__(self, cfg: Dict[str, Any]):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None

        import torch

        self.torch = torch
        self.cfg = cfg
        self.model_name = cfg.get("model_name", "Qwen/Qwen3-8B")
        self.max_length = int(cfg.get("max_length", 1024))
        self.encode_batch_size = int(cfg.get("encode_batch_size", 4))
        self.pooling = cfg.get("pooling", "mean")
        self.normalize_features = bool(cfg.get("normalize_features", True))
        self.prompt_template = cfg.get(
            "prompt_template",
            "Question:\n{question}\n\nModel answer:\n{model_answer}\n",
        )
        cache_dir = cfg.get("cache_dir")

        print(f"[score_llm] loading frozen feature model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.torch_dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        elif self.torch_dtype in {"float16", "fp16"}:
            kwargs["torch_dtype"] = torch.float16
        elif self.torch_dtype in {"bfloat16", "bf16"}:
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.torch_dtype in {"float32", "fp32"}:
            kwargs["torch_dtype"] = torch.float32

        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        if self.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "4-bit loading requires a recent transformers installation with "
                    "BitsAndBytesConfig and bitsandbytes installed."
                )

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs,
        )
        self.model.eval()

    def format_text(self, question: str, model_answer: str) -> str:
        return self.prompt_template.format(question=question, model_answer=model_answer)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cpu"

    def encode_rows(self, rows: List[dict]) -> np.ndarray:
        texts = [self.format_text(r["question"], r["model_answer"]) for r in rows]
        return self.encode_texts(texts)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        all_features = []
        torch = self.torch
        for start in range(0, len(texts), self.encode_batch_size):
            batch = texts[start : start + self.encode_batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = out.hidden_states[-1]
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)

                if self.pooling == "last":
                    lengths = inputs["attention_mask"].sum(dim=1).clamp(min=1) - 1
                    feats = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
                elif self.pooling == "mean":
                    feats = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

                feats = feats.detach().float().cpu().numpy()
                if self.normalize_features:
                    denom = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
                    feats = feats / denom
                all_features.append(feats.astype(np.float32))
        return np.concatenate(all_features, axis=0)
