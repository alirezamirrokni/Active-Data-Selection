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
    text = str(name)
    text = text.replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "none"


def fmt_float(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return safe_name(x)
    return f"{val:g}".replace("-", "m").replace(".", "p")


def _dataset_name(cfg: Dict[str, Any]) -> str:
    data = cfg["data"]
    name = safe_name(data.get("name", "data"))
    split = data.get("split")
    return f"{name}-{safe_name(split)}" if split is not None else name


def _main_llm_name(cfg: Dict[str, Any]) -> str:
    main = cfg["main_llm"]
    return safe_name(main.get("model_name", main.get("provider", "main")))


def _score_model_name(cfg: Dict[str, Any]) -> str:
    score = cfg.get("score_model", {}) or {}
    provider = score.get("provider", "none")
    if provider in {None, "none"}:
        return "none"
    model_name = safe_name(score.get("model_name", provider))
    model_name = model_name.replace("Qwen-Qwen3-", "qwen3-")
    model_name = model_name.replace("gemini-embedding-2", "gemini-emb2")
    return model_name


def _selector_llm_name(cfg: Dict[str, Any]) -> str:
    selector = cfg.get("selector_llm", {}) or {}
    provider = selector.get("provider", "none")
    if provider in {None, "none"}:
        return "none"
    return safe_name(selector.get("model_name", provider))


def _budget_variant(cfg: Dict[str, Any]) -> str:
    policy = cfg["policy"]
    return f"budget{fmt_float(policy.get('budget_per_batch', 0))}"


def _method_params(cfg: Dict[str, Any]) -> list[str]:
    method = cfg.get("method", "method")
    policy = cfg["policy"]
    seed = safe_name(cfg.get("seed", 0))

    if method == "random":
        return [f"seed{seed}"]

    if method == "ours":
        return [
            _score_model_name(cfg),
            f"eps{fmt_float(policy.get('epsilon', 0))}",
            f"alpha{fmt_float(policy.get('alpha_step_size', 0))}",
            f"theta{fmt_float(policy.get('theta_step_size', 0))}",
        ]

    if method == "llm_select":
        return [
            _selector_llm_name(cfg),
            f"seed{seed}",
        ]

    params = [f"seed{seed}"]
    for key in sorted(policy):
        if key in {"budget_per_batch", "cost_variant"}:
            continue
        params.append(f"{safe_name(key)}{fmt_float(policy[key])}")
    return params


def run_name_from_config(cfg: Dict[str, Any]) -> str:
    """Build the method-run CSV/state stem.

    Format:
        {method}_{main_llm}_{dataset}_{budget variant}_{method params}

    The number of batches is intentionally excluded, so a run can be extended
    by increasing data.num_batches and rerunning without changing the output file.
    """
    parts = [
        safe_name(cfg.get("method", "method")),
        _main_llm_name(cfg),
        _dataset_name(cfg),
        _budget_variant(cfg),
    ]
    parts.extend(_method_params(cfg))
    return "_".join(parts)


def generation_cache_name(cfg: Dict[str, Any]) -> str:
    """Build the shared main-LLM generation cache name.

    This cache stores main-model generations keyed by example_id. It is
    independent of the method, score model, number of batches, and budget.
    """
    return f"gen_{_main_llm_name(cfg)}_{_dataset_name(cfg)}.csv"


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
    """Return standard project paths.

    Generation configs only need the shared generation cache. Method configs also
    get a method-specific run CSV and state file. This lets us keep a separate
    configs/generate.yaml with data.max_samples, while method configs use
    data.batch_size/data.num_batches for online sampling.
    """
    out = ensure_dir(cfg.get("output_dir", "outputs"))
    paths = {
        "output_dir": out,
        "generation_cache": out / generation_cache_name(cfg),
    }

    if "method" in cfg and "policy" in cfg:
        run_stem = run_name_from_config(cfg)
        paths.update(
            {
                "run_name": run_stem,
                "run_csv": out / f"{run_stem}.csv",
                "state_json": out / f"{run_stem}_state.json",
            }
        )
    else:
        paths["run_name"] = None

    return paths
