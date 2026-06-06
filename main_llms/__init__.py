from .gemini import GeminiLLM
from .llama import LlamaLLM


def build_main_llm(cfg):
    provider = cfg.get("provider")
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "llama":
        return LlamaLLM(cfg)
    raise ValueError(f"Unknown main_llm.provider: {provider}")
