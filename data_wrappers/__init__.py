from .math500 import Math500Wrapper
from .mmlupro import MMLUProWrapper
from .popqa import PopQAWrapper
from .gpqa import GPQAWrapper


def build_data_wrapper(cfg):
    name = str(cfg.get("name", "")).lower().replace("-", "_")
    if name == "math500":
        return Math500Wrapper(cfg)
    if name in {"popqa", "popqa500"}:
        return PopQAWrapper(cfg)
    if name in {"mmlupro", "mmlu_pro", "mmlupro500", "mmlu_pro500"}:
        return MMLUProWrapper(cfg)
    if name in {"gpqa", "gpqa_diamond", "gpqa_main", "gpqa_extended"}:
        return GPQAWrapper(cfg)
    raise ValueError(f"Unknown data.name: {cfg.get('name')}")
