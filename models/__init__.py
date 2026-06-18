from .gemini import GeminiLLM
from .groq import GroqLLM
from .qwen8b import Qwen8BScoreModel


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "groq":
        return GroqLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")


def build_score_model(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider == "qwen8b":
        return Qwen8BScoreModel(cfg)
    raise ValueError(f"Unknown score_model.provider: {provider}")
