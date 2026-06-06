from .gsm8k import GSM8KWrapper
from .math500 import Math500Wrapper


def build_data_wrapper(cfg):
    name = cfg.get("name")
    if name == "gsm8k":
        return GSM8KWrapper(cfg)
    if name == "math500":
        return Math500Wrapper(cfg)
    raise ValueError(f"Unknown data.name: {name}")
