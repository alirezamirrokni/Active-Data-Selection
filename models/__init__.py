from .gemini import GeminiLLM
from .groq import GroqLLM
from .deepseek import DeepSeekLLM
from .openrouter import OpenRouterLLM
from .qwen import QwenLLM, QwenScoreModel
from .gemini_embedding import GeminiEmbedding2ScoreModel


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "groq":
        return GroqLLM(cfg)
    if provider == "deepseek":
        return DeepSeekLLM(cfg)
    if provider == "openrouter":
        return OpenRouterLLM(cfg)
    if provider in {"qwen", "qwen8b", "qwen_local"}:
        return QwenLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")


def build_score_model(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider in {"qwen", "qwen8b"}:
        return QwenScoreModel(cfg)
    if provider in {"gemini_embedding_2", "gemini-embedding-2", "gemini_embedding"}:
        return GeminiEmbedding2ScoreModel(cfg)
    raise ValueError(f"Unknown score_model.provider: {provider}")
