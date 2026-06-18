from .math500 import Math500Wrapper
from .popqa500 import PopQA500Wrapper


def build_data_wrapper(cfg):
    name = cfg.get("name")
    if name == "math500":
        return Math500Wrapper(cfg)
    if name == "popqa500":
        return PopQA500Wrapper(cfg)
    raise ValueError(f"Unknown data.name: {name}")
