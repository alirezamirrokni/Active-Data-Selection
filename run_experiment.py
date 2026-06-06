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



def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def print_run_summary(df: pd.DataFrame) -> None:
    """Print compact end-of-run metrics from the saved run CSV."""
    if df is None or len(df) == 0:
        print("[summary] no rows available")
        return

    A = df["A"].astype(float)
    selected = df["selected"].astype(float)
    cost = df["cost"].astype(float)
    budget = df["budget"].astype(float)

    n = len(df)
    n_selected = int(selected.sum())
    n_unselected = int(n - n_selected)

    model_correct = int((1.0 - A).sum())
    model_accuracy = _safe_div(model_correct, n)

    type_i = _safe_div(float((selected * (1.0 - A)).sum()), float(selected.sum()))
    type_ii = _safe_div(float(((1.0 - selected) * A).sum()), float((1.0 - selected).sum()))

    spent_total = float((selected * cost).sum())
    budget_total = float(budget.groupby(df["t"]).first().sum()) if "t" in df else float(budget.sum())
    budget_used = _safe_div(spent_total, budget_total)

    print("\n[summary]")
    print(f"  examples          : {n}")
    print(f"  model accuracy    : {model_accuracy:.4f} ({model_correct}/{n})")
    print(f"  selected          : {n_selected}")
    print(f"  unselected        : {n_unselected}")
    print(f"  type-I            : {type_i:.4f}")
    print(f"  type-II           : {type_ii:.4f}")
    print(f"  budget used       : {spent_total:.2f}/{budget_total:.2f} ({100.0 * budget_used:.1f}%)")


def ensure_generations(
    records: List[Dict[str, Any]],
    data_wrapper,
    main_llm,
    cache_path: Path,
    allow_generate: bool = True,
) -> pd.DataFrame:
    """Load or create the shared main-LLM generation cache.

    The cache is method-independent. It stores the main model answer and the
    induced failure label A for each dataset example. If allow_generate=False,
    this function never calls the main LLM; it only validates that all requested
    examples are already cached.
    """
    cache = read_csv_or_empty(cache_path, GEN_COLUMNS)
    target_ids = {int(r["example_id"]) for r in records}
    done_ids = set(cache["example_id"].astype(int).tolist()) if len(cache) else set()
    done_target_ids = done_ids.intersection(target_ids)
    missing = [r for r in records if int(r["example_id"]) not in done_ids]

    if len(cache):
        cache_relevant = cache[cache["example_id"].astype(int).isin(target_ids)]
    else:
        cache_relevant = cache
    cached_seen = len(cache_relevant)
    cached_correct = int((1 - cache_relevant["A"].astype(int)).sum()) if len(cache_relevant) else 0
    cached_acc = _safe_div(cached_correct, cached_seen)

    print(f"[cache] generation cache: {cache_path}")
    print(
        f"[cache] cached_total={len(cache)} loaded_for_this_run={len(done_target_ids)} "
        f"missing={len(missing)} requested={len(records)}"
    )
    if cached_seen:
        print(f"[cache] cached model accuracy on requested rows={cached_acc:.4f} ({cached_correct}/{cached_seen})")

    if missing and not allow_generate:
        raise RuntimeError(
            f"Generation cache is incomplete: {len(missing)} missing examples. "
            f"Run `python generate_cache.py --config <config>` first, or rerun without `--no_generate`."
        )

    rows = cache.to_dict("records") if len(cache) else []
    if not missing:
        return cache

    if main_llm is None:
        raise RuntimeError("main_llm is required because generation cache has missing examples.")

    seen = cached_seen
    correct = cached_correct
    pbar = tqdm(missing, desc="main LLM generations", dynamic_ncols=True)
    for rec in pbar:
        prompt = data_wrapper.build_prompt(rec["question"])
        model_answer = main_llm.generate(prompt)
        pred_answer = data_wrapper.parse_prediction(model_answer)
        A = data_wrapper.failure_label(pred_answer, rec["gold_final"])
        seen += 1
        correct += int(A == 0)
        running_acc = _safe_div(correct, seen)
        pbar.set_postfix(acc=f"{running_acc:.3f}", correct=f"{correct}/{seen}")
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


def run(cfg_path: str, reset: bool = False, reset_generations: bool = False, no_generate: bool = False) -> None:
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

    main_llm = None
    if not no_generate:
        # The main LLM is built lazily only when generation is allowed. This
        # lets method-only runs use an existing cache without requiring an API key.
        main_llm = build_main_llm(cfg["main_llm"])
    gen_cache = ensure_generations(
        records,
        data_wrapper,
        main_llm,
        paths["generation_cache"],
        allow_generate=not no_generate,
    )
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

    method_pbar = tqdm(batch_starts, desc=cfg["run_name"], dynamic_ncols=True)
    for start in method_pbar:
        end = min(start + batch_size, n)
        t = start // batch_size
        batch_records = records[start:end]
        ids = [int(r["example_id"]) for r in batch_records]

        if all(i in done_ids for i in ids):
            partial_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS) if all_rows else run_df
            if len(partial_df):
                running_acc = _safe_div(int((1 - partial_df["A"].astype(int)).sum()), len(partial_df))
                method_pbar.set_postfix(acc=f"{running_acc:.3f}")
            continue

        batch_df = make_batch_rows(cfg, gen_by_id, batch_records, t)
        out_batch, new_method_state = method.process_batch(batch_df, t=t)
        out_batch = out_batch[RUN_COLUMNS]

        # Replace any partial stale rows from this batch, then append the new complete batch.
        existing = pd.DataFrame(all_rows, columns=RUN_COLUMNS) if all_rows else pd.DataFrame(columns=RUN_COLUMNS)
        existing = existing[~existing["example_id"].astype(str).isin([str(i) for i in ids])]
        all_rows = existing.to_dict("records") + out_batch.to_dict("records")
        current_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS)
        write_csv_atomic(current_df, paths["run_csv"])

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
        running_correct = int((1 - current_df["A"].astype(int)).sum())
        running_acc = _safe_div(running_correct, len(current_df))
        method_pbar.set_postfix(acc=f"{running_acc:.3f}", rows=len(current_df))
        print(
            f"[batch {t:03d}] selected={n_sel:3d} spent={spent:.1f} "
            f"type-I={type_i:.3f} model-acc={running_acc:.3f} ({running_correct}/{len(current_df)})"
        )

    final_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS)
    print_run_summary(final_df)
    print(f"[done] wrote {paths['run_csv']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--reset", action="store_true", help="Delete this config's run CSV/state before running.")
    parser.add_argument("--reset_generations", action="store_true", help="Delete shared main-LLM generation cache too.")
    parser.add_argument(
        "--no_generate",
        action="store_true",
        help="Never call the main LLM. Require all requested examples to already exist in the generation cache.",
    )
    args = parser.parse_args()
    run(args.config, reset=args.reset, reset_generations=args.reset_generations, no_generate=args.no_generate)


if __name__ == "__main__":
    main()
