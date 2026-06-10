from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GeminiEmbedding2Config:
    provider: str = "gemini_embedding_2"
    model_name: str = "gemini-embedding-2"
    output_dimensionality: int = 768
    encode_batch_size: int = 8
    normalize_features: bool = True
    min_seconds_between_calls: float = 0.25
    retry_attempts: int = 8
    retry_sleep: float = 2.0
    prompt_template: str = (
        "Represent this prompt-response pair for predicting whether the model answer "
        "will require human correction.\n\n"
        "Problem:\n{question}\n\nModel answer:\n{model_answer}"
    )


class GeminiEmbedding2ScoreModel:
    """Gemini Embedding 2 feature extractor used by the online score model."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise ImportError("Install google-genai from requirements.txt first.") from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. "
                "Set it before using Gemini Embedding 2 as a score model."
            )

        self.cfg = GeminiEmbedding2Config(**cfg)
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self._last_call_time = 0.0
        self._cache: Dict[str, np.ndarray] = {}

        print(
            "[score_model] using Gemini Embedding 2 "
            f"model={self.cfg.model_name} dim={self.cfg.output_dimensionality} "
            f"batch_size={self.cfg.encode_batch_size}"
        )

    def format_text(self, question: str, model_answer: str) -> str:
        return self.cfg.prompt_template.format(
            question=str(question),
            model_answer=str(model_answer),
        )

    def encode_rows(self, rows: List[dict]) -> np.ndarray:
        texts = [
            self.format_text(
                question=row.get("question", ""),
                model_answer=row.get("model_answer", ""),
            )
            for row in rows
        ]
        return self.encode_texts(texts)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

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
            r"retry after\s*([0-9.]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1)) + 2.0
                except Exception:
                    pass
        return default

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, int(self.cfg.output_dimensionality)), dtype=np.float32)

        out: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is None:
                missing_indices.append(i)
                missing_texts.append(text)
            else:
                out[i] = cached

        batch_size = max(1, int(self.cfg.encode_batch_size))
        for start in range(0, len(missing_texts), batch_size):
            batch_texts = missing_texts[start : start + batch_size]
            batch_indices = missing_indices[start : start + batch_size]
            embeddings = self._embed_batch(batch_texts)
            for idx, text, emb in zip(batch_indices, batch_texts, embeddings):
                self._cache[self._cache_key(text)] = emb
                out[idx] = emb

        arr = np.stack([x for x in out if x is not None], axis=0).astype(np.float32)
        return arr

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                contents = [
                    self.types.Content(
                        parts=[self.types.Part.from_text(text=text)]
                    )
                    for text in texts
                ]
                result = self.client.models.embed_content(
                    model=self.cfg.model_name,
                    contents=contents,
                    config=self.types.EmbedContentConfig(
                        output_dimensionality=int(self.cfg.output_dimensionality)
                    ),
                )
                self._last_call_time = time.time()
                embeddings = [
                    np.asarray(emb.values, dtype=np.float32)
                    for emb in result.embeddings
                ]
                if len(embeddings) != len(texts):
                    # Defensive fallback: if the API returns an aggregated embedding,
                    # embed each text separately.
                    return [self._embed_one(text) for text in texts]
                return [self._normalize(e) for e in embeddings]
            except Exception as exc:
                last_err = exc
                base_wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "rate" in str(exc).lower():
                    wait = self._retry_after_seconds(exc, default=max(30.0, base_wait))
                else:
                    wait = base_wait
                print(
                    f"[score_model:gemini_embedding_2] attempt {attempt} failed: {exc}. "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"Gemini embedding call failed after retries: {last_err}")

    def _embed_one(self, text: str) -> np.ndarray:
        self._throttle()
        result = self.client.models.embed_content(
            model=self.cfg.model_name,
            contents=text,
            config=self.types.EmbedContentConfig(
                output_dimensionality=int(self.cfg.output_dimensionality)
            ),
        )
        self._last_call_time = time.time()
        emb = np.asarray(result.embeddings[0].values, dtype=np.float32)
        return self._normalize(emb)

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_features:
            return emb.astype(np.float32)
        denom = np.linalg.norm(emb) + 1e-12
        return (emb / denom).astype(np.float32)
