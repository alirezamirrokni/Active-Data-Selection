from .math500 import Math500Wrapper
from .triviaqa500 import TriviaQA500Wrapper


def build_data_wrapper(cfg):
    name = cfg.get("name")
    if name == "math500":
        return Math500Wrapper(cfg)
    if name == "triviaqa500":
        return TriviaQA500Wrapper(cfg)
    raise ValueError(f"Unknown data.name: {name}")
