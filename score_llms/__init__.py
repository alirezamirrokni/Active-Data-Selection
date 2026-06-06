from .qwen3 import Qwen3ScoreLLM


def build_score_llm(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider == "qwen3":
        return Qwen3ScoreLLM(cfg)
    raise ValueError(f"Unknown score_llm.provider: {provider}")
