from .gemini import GeminiLLM
from .groq import GroqLLM


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "groq":
        return GroqLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")
