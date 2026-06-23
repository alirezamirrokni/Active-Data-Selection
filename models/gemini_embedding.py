from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GeminiEmbeddingConfig:
    provider: str = "gemini_embedding_2"
    model_name: str = "gemini-embedding-2"
    output_dimensionality: int = 3072
    task_type: str = "SEMANTIC_SIMILARITY"
    title: Optional[str] = None
    max_chars: int = 12000
    encode_batch_size: int = 16
    normalize_features: bool = True
    request_timeout: int = 120
    retry_attempts: int = 8
    retry_sleep: float = 2.0
    min_seconds_between_calls: float = 0.5
    prompt_template: Optional[str] = None
    dataset_name: str = "math500"


class GeminiEmbedding2ScoreModel:
    """Gemini Embedding 2 feature model used by the EDIT linear probe.

    The model maps each prompt-response pair Z=(X, \u007eY) to a dense feature vector.
    These vectors can be used as feature representations in the `ours` method.
    """

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise ImportError("Install google-genai from requirements.txt first.") from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Add it to .env or export it."
            )

        cfg = dict(cfg)
        cfg.setdefault("provider", "gemini_embedding_2")
        cfg.setdefault("model_name", "gemini-embedding-2")
        cfg.setdefault("dataset_name", "math500")
        self.cfg = GeminiEmbeddingConfig(**cfg)
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self._last_call_time = 0.0
        self.cache: Dict[str, np.ndarray] = {}

        dataset_name = str(self.cfg.dataset_name).lower().replace("-", "_")
        if self.cfg.prompt_template:
            self.prompt_template = self.cfg.prompt_template
        elif dataset_name in {"popqa", "popqa500"}:
            self.prompt_template = "Question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        elif dataset_name in {"mmlupro", "mmlu_pro", "mmlupro500", "mmlu_pro500"}:
            self.prompt_template = "Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        else:
            self.prompt_template = "Problem:\n\n{question}\n\nModel answer:\n\n{model_answer}"

        print(
            "[score_model:gemini_embedding_2] using "
            f"model={self.cfg.model_name} dim={self.cfg.output_dimensionality} "
            f"task_type={self.cfg.task_type}"
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
            r"retry after\s*([0-9.]+)s",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) + 2.0
                except Exception:
                    pass
        return default

    def _row_text(self, row: Dict[str, Any]) -> str:
        text = self.prompt_template.format(
            question=str(row.get("question", "")),
            model_answer=str(row.get("model_answer", "")),
        )
        max_chars = int(self.cfg.max_chars)
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max_chars - 20] + " ... [truncated]"
        return text

    def _cache_key(self, text: str) -> str:
        payload = f"{self.cfg.model_name}\n{self.cfg.output_dimensionality}\n{self.cfg.task_type}\n{text}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _embedding_values(embedding: Any) -> List[float]:
        values = getattr(embedding, "values", None)
        if values is not None:
            return list(values)
        if isinstance(embedding, dict):
            if "values" in embedding:
                return list(embedding["values"])
            if "embedding" in embedding:
                return list(embedding["embedding"])
        return list(embedding)

    def _embed_texts_api(self, texts: List[str]) -> List[np.ndarray]:
        config_kwargs: Dict[str, Any] = {
            "task_type": self.cfg.task_type,
            "output_dimensionality": int(self.cfg.output_dimensionality),
        }
        if self.cfg.title:
            config_kwargs["title"] = self.cfg.title

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                response = self.client.models.embed_content(
                    model=self.cfg.model_name,
                    contents=texts,
                    config=self.types.EmbedContentConfig(**config_kwargs),
                )
                self._last_call_time = time.time()

                embeddings = getattr(response, "embeddings", None)
                if embeddings is None and isinstance(response, dict):
                    embeddings = response.get("embeddings")
                if embeddings is None:
                    raise RuntimeError(f"No embeddings returned by Gemini response: {response}")

                vectors = [
                    np.asarray(self._embedding_values(emb), dtype=np.float32)
                    for emb in embeddings
                ]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Gemini returned {len(vectors)} embeddings for {len(texts)} texts."
                    )
                return vectors

            except Exception as exc:
                last_err = exc
                base_wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "rate" in str(exc).lower():
                    wait = self._retry_after_seconds(exc, default=max(60.0, base_wait))
                else:
                    wait = base_wait
                print(
                    f"[score_model:gemini_embedding_2] attempt {attempt} failed: {exc}. "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini Embedding 2 call failed after retries: {last_err}")

    def encode_rows(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        texts = [self._row_text(row) for row in rows]
        keys = [self._cache_key(text) for text in texts]

        missing_texts: List[str] = []
        missing_keys: List[str] = []
        for key, text in zip(keys, texts):
            if key not in self.cache:
                missing_keys.append(key)
                missing_texts.append(text)

        batch_size = max(1, int(self.cfg.encode_batch_size))
        for start in range(0, len(missing_texts), batch_size):
            batch_texts = missing_texts[start : start + batch_size]
            batch_keys = missing_keys[start : start + batch_size]
            vectors = self._embed_texts_api(batch_texts)
            for key, vec in zip(batch_keys, vectors):
                self.cache[key] = vec

        feats = np.stack([self.cache[key] for key in keys], axis=0).astype(np.float32)
        if self.cfg.normalize_features:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.maximum(norms, 1e-12)
        return feats
