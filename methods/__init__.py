from .llm_select import LLMSelect
from .ours import OursSelection
from .random import RandomSelection


def build_method(cfg, score_model=None, state=None):
    method = cfg.get("method")
    if method == "ours":
        return OursSelection(cfg, score_model=score_model, state=state)
    if method == "random":
        return RandomSelection(cfg, score_model=score_model, state=state)
    if method == "llm_select":
        return LLMSelect(cfg, score_model=score_model, state=state)
    raise ValueError(f"Unknown method: {method}")
