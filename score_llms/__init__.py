from .qwen3_8b import Qwen3EightBScoreLLM


def build_score_llm(cfg):
    provider = cfg.get("provider", "none")
    if provider in {None, "none"}:
        return None
    if provider == "qwen3_8b":
        return Qwen3EightBScoreLLM(cfg)
    raise ValueError(f"Unknown score_llm.provider: {provider}")
