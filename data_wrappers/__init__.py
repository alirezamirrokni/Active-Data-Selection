from .math500 import Math500Wrapper
from .mmlupro import MMLUProWrapper
from .popqa import PopQAWrapper


def build_data_wrapper(cfg):
    name = str(cfg.get("name", "")).lower().replace("-", "_")
    if name == "math500":
        return Math500Wrapper(cfg)
    if name in {"popqa", "popqa500"}:
        return PopQAWrapper(cfg)
    if name in {"mmlupro", "mmlu_pro", "mmlupro500", "mmlu_pro500"}:
        return MMLUProWrapper(cfg)
    raise ValueError(f"Unknown data.name: {cfg.get('name')}")
