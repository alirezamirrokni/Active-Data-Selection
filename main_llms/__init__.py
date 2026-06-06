from .dummy import DummyLLM
from .gemini import GeminiLLM


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "dummy":
        return DummyLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")
