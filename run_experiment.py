import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from data_wrappers import build_data_wrapper
from main_llms import build_main_llm
from methods import build_method
from methods.costs import compute_cost
from score_llms import build_score_llm
from utils import load_json, load_yaml, project_paths, read_csv_or_empty, save_json_atomic, write_csv_atomic


GEN_COLUMNS = [
    "example_id",
    "question",
    "gold_answer",
    "gold_final",
    "model_answer",
    "pred_answer",
    "A",
]

RUN_COLUMNS = [
    "config",
    "method",
    "t",
    "example_id",
    "question",
    "gold_answer",
    "gold_final",
    "model_answer",
    "pred_answer",
    "A",
    "cost",
    "eta",
    "alpha",
    "beta",
    "selected",
    "budget",
    "epsilon",
    "main_llm_provider",
    "main_llm",
    "score_llm_provider",
    "score_llm",
]


def ensure_generations(records: List[Dict[str, Any]], data_wrapper, main_llm, cache_path: Path) -> pd.DataFrame:
    cache = read_csv_or_empty(cache_path, GEN_COLUMNS)
    done_ids = set(cache["example_id"].astype(int).tolist()) if len(cache) else set()
    missing = [r for r in records if int(r["example_id"]) not in done_ids]

    print(f"[cache] generation cache: {cache_path}")
    print(f"[cache] loaded={len(done_ids)} missing={len(missing)} total={len(records)}")

    rows = cache.to_dict("records") if len(cache) else []
    if not missing:
        return cache

    for rec in tqdm(missing, desc="main LLM generations", dynamic_ncols=True):
        prompt = data_wrapper.build_prompt(rec["question"])
        model_answer = main_llm.generate(prompt)
        pred_answer = data_wrapper.parse_prediction(model_answer)
        A = data_wrapper.failure_label(pred_answer, rec["gold_final"])
        rows.append(
            {
                "example_id": rec["example_id"],
                "question": rec["question"],
                "gold_answer": rec["gold_answer"],
                "gold_final": rec["gold_final"],
                "model_answer": model_answer,
                "pred_answer": pred_answer,
                "A": int(A),
            }
        )
        write_csv_atomic(pd.DataFrame(rows, columns=GEN_COLUMNS), cache_path)

    return pd.DataFrame(rows, columns=GEN_COLUMNS)


def make_batch_rows(
    cfg: Dict[str, Any],
    gen_by_id: pd.DataFrame,
    records: List[Dict[str, Any]],
    t: int,
) -> pd.DataFrame:
    rows = []
    policy = cfg["policy"]
    score_cfg = cfg.get("score_llm", {})
    for rec in records:
        cached = gen_by_id.loc[int(rec["example_id"])].to_dict()
        row = {
            "config": cfg.get("run_name", cfg["method"]),
            "method": cfg["method"],
            "t": t,
            "example_id": int(rec["example_id"]),
            "question": cached["question"],
            "gold_answer": cached["gold_answer"],
            "gold_final": cached["gold_final"],
            "model_answer": cached["model_answer"],
            "pred_answer": cached["pred_answer"],
            "A": int(cached["A"]),
            "cost": 1.0,
            "eta": float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "selected": 0,
            "budget": float(policy["budget_per_batch"]),
            "epsilon": float(policy["epsilon"]),
            "main_llm_provider": cfg["main_llm"].get("provider"),
            "main_llm": cfg["main_llm"].get("model_name"),
            "score_llm_provider": score_cfg.get("provider", "none"),
            "score_llm": score_cfg.get("model_name", "none"),
        }
        row["cost"] = compute_cost(row, policy.get("cost_variant", "constant"))
        rows.append(row)
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def run(cfg_path: str, reset: bool = False, reset_generations: bool = False) -> None:
    load_dotenv()
    cfg = load_yaml(cfg_path)
    paths = project_paths(cfg)
    cfg["run_name"] = paths["run_name"]

    if reset:
        for p in [paths["run_csv"], paths["state_json"]]:
            if p.exists():
                p.unlink()
    if reset_generations and paths["generation_cache"].exists():
        paths["generation_cache"].unlink()

    print(f"[run] name={cfg['run_name']} method={cfg['method']}")
    print(f"[run] run csv={paths['run_csv']}")

    data_wrapper = build_data_wrapper(cfg["data"])
    records = data_wrapper.load_records()
    batch_size = int(cfg["data"].get("batch_size", 20))

    main_llm = build_main_llm(cfg["main_llm"])
    gen_cache = ensure_generations(records, data_wrapper, main_llm, paths["generation_cache"])
    gen_by_id = gen_cache.set_index("example_id")

    run_df = read_csv_or_empty(paths["run_csv"], RUN_COLUMNS)
    done_ids = set(run_df["example_id"].astype(int).tolist()) if len(run_df) else set()
    if done_ids:
        print(f"[resume] loaded {len(done_ids)} completed rows")

    state = load_json(paths["state_json"], default={})
    method_state = state.get("method_state", {})

    score_llm = None
    if cfg.get("method") == "ours":
        score_llm = build_score_llm(cfg.get("score_llm", {"provider": "none"}))
    method = build_method(cfg, score_llm=score_llm, state=method_state)

    all_rows = run_df.to_dict("records") if len(run_df) else []
    n = len(records)
    batch_starts = list(range(0, n, batch_size))

    for start in tqdm(batch_starts, desc=cfg["run_name"], dynamic_ncols=True):
        end = min(start + batch_size, n)
        t = start // batch_size
        batch_records = records[start:end]
        ids = [int(r["example_id"]) for r in batch_records]

        if all(i in done_ids for i in ids):
            continue

        batch_df = make_batch_rows(cfg, gen_by_id, batch_records, t)
        out_batch, new_method_state = method.process_batch(batch_df, t=t)
        out_batch = out_batch[RUN_COLUMNS]

        # Replace any partial stale rows from this batch, then append the new complete batch.
        existing = pd.DataFrame(all_rows, columns=RUN_COLUMNS) if all_rows else pd.DataFrame(columns=RUN_COLUMNS)
        existing = existing[~existing["example_id"].astype(str).isin([str(i) for i in ids])]
        all_rows = existing.to_dict("records") + out_batch.to_dict("records")
        write_csv_atomic(pd.DataFrame(all_rows, columns=RUN_COLUMNS), paths["run_csv"])

        done_ids.update(ids)
        save_json_atomic(
            {
                "config": cfg.get("run_name", cfg["method"]),
                "next_batch_start": end,
                "method_state": new_method_state,
            },
            paths["state_json"],
        )

        n_sel = int(out_batch["selected"].sum())
        type_i = 0.0 if n_sel == 0 else float(((out_batch["selected"] * (1 - out_batch["A"])).sum()) / n_sel)
        spent = float((out_batch["selected"] * out_batch["cost"]).sum())
        print(f"[batch {t:03d}] selected={n_sel:3d} spent={spent:.1f} type-I={type_i:.3f}")

    print(f"[done] wrote {paths['run_csv']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--reset", action="store_true", help="Delete this config's run CSV/state before running.")
    parser.add_argument("--reset_generations", action="store_true", help="Delete shared main-LLM generation cache too.")
    args = parser.parse_args()
    run(args.config, reset=args.reset, reset_generations=args.reset_generations)


if __name__ == "__main__":
    main()
