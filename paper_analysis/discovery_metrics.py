from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_analysis.common import (
    DATASET_SPECS,
    METHOD_ORDER,
    MODEL_LABELS,
    compute_metrics,
    load_main_config,
    locate_main_run_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the complementary discovery metrics reported in the paper "
            "from existing main-run CSVs. The outputs/ tree is read-only."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("llama", "qwen"),
        default=("llama", "qwen"),
        help="Response models to include (default: llama qwen).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("paper_artifacts/discovery_metrics"),
        help="Directory for generated CSV/LaTeX artifacts (never outputs/).",
    )
    return parser.parse_args()


def _rank_format(
    group: pd.DataFrame, metric: str, *, higher_is_better: bool, integer: bool = False
) -> dict[str, str]:
    ranked = group[["Method", metric]].copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="raise")
    ranked = ranked.sort_values(metric, ascending=not higher_is_better)
    result: dict[str, str] = {}
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        value = int(round(row[metric])) if integer else float(row[metric])
        text = f"{value:d}" if integer else f"{value:.3f}"
        if rank == 1:
            text = rf"\textbf{{{text}}}"
        elif rank == 2:
            text = rf"\underline{{{text}}}"
        result[str(row["Method"])] = text
    return result


def build_metrics(models: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for model_key in models:
        for spec in DATASET_SPECS:
            cfg = load_main_config(model_key, spec)
            run_files = locate_main_run_files(cfg)
            print(f"{MODEL_LABELS[model_key]:15s} | {spec.label:10s} | {list(run_files)}")
            for method in METHOD_ORDER:
                path = run_files.get(method)
                if path is None:
                    continue
                metrics = compute_metrics(pd.read_csv(path))
                metrics.update(
                    {
                        "Model": MODEL_LABELS[model_key],
                        "Dataset": spec.label,
                        "Method": method,
                        "source_csv": str(path),
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def build_paper_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, dataset), group in metrics.groupby(
        ["Model", "Dataset"], sort=False, observed=True
    ):
        recall = _rank_format(group, "edit_recall", higher_is_better=True)
        missed = _rank_format(
            group, "absolute_missed_edits", higher_is_better=False, integer=True
        )
        precision = _rank_format(group, "edit_precision", higher_is_better=True)
        for _, row in group.iterrows():
            method = str(row["Method"])
            rows.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "Method": method,
                    r"Edit Recall $\uparrow$": recall[method],
                    r"Missed Edits $\downarrow$": missed[method],
                    r"Edit Precision $\uparrow$": precision[method],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics(list(args.models))
    metrics.to_csv(artifact_dir / "discovery_metrics_full.csv", index=False)

    paper = build_paper_table(metrics)
    paper.to_csv(artifact_dir / "discovery_metrics_paper.csv", index=False)
    (artifact_dir / "discovery_metrics_paper.tex").write_text(
        paper.to_latex(index=False, escape=False, column_format="lllccc"),
        encoding="utf-8",
    )

    print("\nComplementary discovery metrics:")
    print(
        metrics[
            [
                "Model",
                "Dataset",
                "Method",
                "edit_recall",
                "absolute_missed_edits",
                "edit_precision",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
