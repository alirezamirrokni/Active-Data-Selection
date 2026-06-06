from typing import Any, Dict, List

import numpy as np


class Qwen3ScoreLLM:
    """Frozen Qwen3-family feature model used only for the online score \tilde{eta}."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ImportError("Install torch and transformers from requirements.txt first.") from exc

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
            "cache_dir": cache_dir,
            "device_map": cfg.get("device_map", "auto"),
            "trust_remote_code": True,
        }
        torch_dtype = cfg.get("torch_dtype", "auto")
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        else:
            kwargs["torch_dtype"] = "auto"

        if bool(cfg.get("load_in_4bit", False)):
            kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
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
