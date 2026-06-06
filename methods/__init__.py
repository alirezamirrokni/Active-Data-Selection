from .ours import OursSelection
from .random import RandomSelection


def build_method(cfg, score_llm=None, state=None):
    method = cfg.get("method")
    if method == "ours":
        return OursSelection(cfg, score_llm=score_llm, state=state)
    if method == "random":
        return RandomSelection(cfg, score_llm=score_llm, state=state)
    raise ValueError(f"Unknown method: {method}")
