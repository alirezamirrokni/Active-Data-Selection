from .dummy import DummyLLM
from .gemini import GeminiLLM
from .llama import LlamaLLM


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider in {"llama", "llama_3_3_70b_versatile", "llama-3.3-70b-versatile"}:
        return LlamaLLM(cfg)
    if provider == "dummy":
        return DummyLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")
