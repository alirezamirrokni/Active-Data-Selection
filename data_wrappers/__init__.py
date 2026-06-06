from .gsm8k import GSM8KWrapper


def build_data_wrapper(cfg):
    name = cfg.get("name")
    if name == "gsm8k":
        return GSM8KWrapper(cfg)
    raise ValueError(f"Unknown data.name: {name}")
