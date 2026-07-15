from .gemini import GeminiLLM
from .groq import GroqLLM
from .deepseek import DeepSeekLLM
from .openrouter import OpenRouterLLM
from .gemini_embedding import GeminiEmbedding2ScoreModel
from .minilm import MiniLMScoreModel
from .qwen3_embedding import Qwen3ScoreModel


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
    raise ValueError(f"Unknown main_llm.provider: {provider}")


def build_score_model(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider in {"minilm", "all_minilm_l6_v2", "all-MiniLM-L6-v2", "sentence_transformers_minilm"}:
        return MiniLMScoreModel(cfg)
    if provider in {"qwen3", "qwen3_embedding", "qwen3-embedding", "qwen_embedding"}:
        return Qwen3ScoreModel(cfg)
    if provider in {"gemini_embedding_2", "gemini-embedding-2", "gemini_embedding"}:
        return GeminiEmbedding2ScoreModel(cfg)
    raise ValueError(f"Unknown score_model.provider: {provider}")
