import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(name: Any) -> str:
    """Convert config values into short filesystem-safe fragments."""
    text = str(name)
    text = text.replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "none"


def fmt_float(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return safe_name(x)
    # Compact but stable: 0.10 -> 0p1, 5.0 -> 5.
    s = f"{val:g}".replace("-", "m").replace(".", "p")
    return s


def run_name_from_config(cfg: Dict[str, Any]) -> str:
    """Build the CSV/state stem from the actual experimental configuration."""
    data = cfg["data"]
    main = cfg["main_llm"]
    score = cfg.get("score_llm", {"provider": "none", "model_name": "none"}) or {}
    policy = cfg["policy"]

    parts = [
        safe_name(cfg["method"]),
        f"data-{safe_name(data.get('name', 'data'))}-{safe_name(data.get('split', 'split'))}",
        f"n{safe_name(data.get('max_examples', 'all'))}",
        f"main-{safe_name(main.get('provider', 'main'))}-{safe_name(main.get('model_name', 'model'))}",
        f"prompt-{safe_name(main.get('prompt_version', 'default'))}",
        f"score-{safe_name(score.get('provider', 'none'))}-{safe_name(score.get('model_name', 'none'))}",
        f"cost-{safe_name(policy.get('cost_variant', 'constant'))}",
        f"eps{fmt_float(policy.get('epsilon', 0))}",
        f"budget{fmt_float(policy.get('budget_per_batch', 0))}",
        f"seed{safe_name(cfg.get('seed', 0))}",
    ]
    return "__".join(parts)


def generation_cache_name(cfg: Dict[str, Any]) -> str:
    data = cfg["data"]
    main = cfg["main_llm"]
    parts = [
        "gen",
        safe_name(data.get("name", "data")),
        safe_name(data.get("split", "split")),
        f"n{safe_name(data.get('max_examples', 'all'))}",
        safe_name(main.get("provider", "main")),
        safe_name(main.get("model_name", "model")),
        f"prompt{safe_name(main.get('prompt_version', 'default'))}",
        f"temp{fmt_float(main.get('temperature', 0))}",
    ]
    return "__".join(parts) + ".csv"


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_csv_or_empty(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def load_json(path: str | Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def project_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    out = ensure_dir(cfg.get("output_dir", "outputs"))
    run_stem = run_name_from_config(cfg)
    return {
        "output_dir": out,
        "run_name": run_stem,
        "generation_cache": out / generation_cache_name(cfg),
        "run_csv": out / f"{run_stem}.csv",
        "state_json": out / f"{run_stem}_state.json",
    }
