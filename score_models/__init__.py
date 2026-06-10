from .gemini_embedding_2 import GeminiEmbedding2ScoreModel
from .qwen3 import Qwen3ScoreModel


def build_score_model(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider == "gemini_embedding_2":
        return GeminiEmbedding2ScoreModel(cfg)
    if provider == "qwen3":
        return Qwen3ScoreModel(cfg)
    raise ValueError(f"Unknown score_model.provider: {provider}")
