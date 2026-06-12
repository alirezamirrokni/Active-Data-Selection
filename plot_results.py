import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLOT_METRICS = {
    "type_ii": {
        "column": "Cum. Type-II",
        "ylabel": "Type-II",
        "filename": "type_ii",
    },
    "type_i": {
        "column": "Cum. Type-I",
        "ylabel": "Type-I",
        "filename": "type_i",
    },
    "budget": {
        "column": "Cum. Budget",
        "ylabel": "Budget",
        "filename": "budget",
    },
}


def collect_run_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []

    for p in paths:
        path = Path(p)
        csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]

        for csv in csvs:
            if csv.name.startswith("gen_") or csv.name.endswith("_metrics.csv"):
                continue
            df = pd.read_csv(csv)
            df["run_file"] = csv.stem
            frames.append(df)

    if not frames:
        raise RuntimeError("No method run CSV files found.")

    return pd.concat(frames, ignore_index=True)


def pretty_method(name: str) -> str:
    mapping = {
        "ours": "Ours",
        "random": "Random",
        "llm_select": "LLM-Select",
    }
    return mapping.get(str(name), str(name))


def safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def summarize_per_round(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["config", "method", "run_file", "t"]

    for (config, method, run_file, t), g in df.groupby(group_cols, sort=True):
        selected = g["selected"].astype(int)
        A = g["A"].astype(int)
        cost = g["cost"].astype(float)

        n_sel = int(selected.sum())
        n_unsel = int((1 - selected).sum())

        selected_correct = float((selected * (1 - A)).sum())
        unselected_wrong = float(((1 - selected) * A).sum())

        rows.append(
            {
                "config": config,
                "method": method,
                "run_file": run_file,
                "Method": pretty_method(method),
                "Round": int(t),
                "Type-I": safe_div(selected_correct, n_sel),
                "Type-II": safe_div(unselected_wrong, n_unsel),
                "Budget": float((selected * cost).sum()),
                "Limit": float(g["budget"].iloc[0]),
                "Selected": n_sel,
                "Unselected": n_unsel,
                "Selected correct": selected_correct,
                "Unselected wrong": unselected_wrong,
            }
        )

    return pd.DataFrame(rows)


def add_cumulative_metrics(round_df: pd.DataFrame) -> pd.DataFrame:
    round_df = round_df.sort_values(
        ["config", "method", "run_file", "Round"]
    ).reset_index(drop=True)

    group_cols = ["config", "method", "run_file"]
    round_index = round_df.groupby(group_cols).cumcount() + 1

    # Running averages of the per-round ratios. These match the paper's
    # empirical average-over-rounds definition.
    round_df["Cum. Type-I"] = round_df.groupby(group_cols)["Type-I"].cumsum() / round_index
    round_df["Cum. Type-II"] = round_df.groupby(group_cols)["Type-II"].cumsum() / round_index
    round_df["Cum. Budget"] = round_df.groupby(group_cols)["Budget"].cumsum() / round_index

    # Pooled ratios are useful diagnostics: they pool all selected/unselected
    # examples observed up to the current round.
    round_df["Cum. Selected"] = round_df.groupby(group_cols)["Selected"].cumsum()
    round_df["Cum. Unselected"] = round_df.groupby(group_cols)["Unselected"].cumsum()
    round_df["Cum. Selected correct"] = round_df.groupby(group_cols)["Selected correct"].cumsum()
    round_df["Cum. Unselected wrong"] = round_df.groupby(group_cols)["Unselected wrong"].cumsum()

    round_df["Pooled Type-I"] = round_df.apply(
        lambda r: safe_div(r["Cum. Selected correct"], r["Cum. Selected"]),
        axis=1,
    )
    round_df["Pooled Type-II"] = round_df.apply(
        lambda r: safe_div(r["Cum. Unselected wrong"], r["Cum. Unselected"]),
        axis=1,
    )

    return round_df


def maybe_add_epsilon(metrics: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if "epsilon" not in raw_df.columns:
        metrics["epsilon"] = pd.NA
        return metrics

    eps = raw_df[["config", "method", "run_file", "epsilon"]].drop_duplicates()
    eps = eps.dropna(subset=["epsilon"])
    if eps.empty:
        metrics["epsilon"] = pd.NA
        return metrics

    return metrics.merge(eps, on=["config", "method", "run_file"], how="left")


def set_paper_style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_metric_plot(metrics: pd.DataFrame, metric_key: str, out_dir: Path) -> None:
    meta = PLOT_METRICS[metric_key]
    y = meta["column"]

    fig, ax = plt.subplots(figsize=(4.35, 2.85))
    sns.lineplot(
        data=metrics,
        x="Round",
        y=y,
        hue="Method",
        linewidth=2.0,
        errorbar=None,
        ax=ax,
    )

    if metric_key == "type_i" and metrics["epsilon"].notna().any():
        epsilon = float(metrics["epsilon"].dropna().iloc[0])
        ax.axhline(
            epsilon,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label=r"$\epsilon$",
        )

    if metric_key == "budget":
        limit = float(metrics["Limit"].iloc[0])
        ax.axhline(
            limit,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label="Limit",
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(meta["ylabel"])
    ax.set_title("")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title=None, loc="best")

    fig.tight_layout(pad=0.35)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{meta['filename']}.{ext}", bbox_inches="tight")
    plt.close(fig)


def print_final_summary(metrics: pd.DataFrame) -> None:
    final_rows = (
        metrics.sort_values(["config", "method", "run_file", "Round"])
        .groupby(["config", "method", "run_file"], as_index=False)
        .tail(1)
    )

    cols = [
        "Method",
        "Round",
        "Cum. Type-I",
        "Cum. Type-II",
        "Pooled Type-I",
        "Pooled Type-II",
        "Cum. Budget",
        "Limit",
    ]

    print()
    print("[plot] final cumulative summary")
    print(final_rows[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs"],
        help="Run CSVs or output directories.",
    )
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    raw_df = collect_run_csvs(args.runs)
    per_round = summarize_per_round(raw_df)
    metrics = add_cumulative_metrics(per_round)
    metrics = maybe_add_epsilon(metrics, raw_df)

    per_round_path = out_dir / "per_round_metrics.csv"
    metrics_path = out_dir / "summary_metrics.csv"
    per_round.to_csv(per_round_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(f"[plot] wrote {per_round_path}")
    print(f"[plot] wrote {metrics_path}")

    for key in ["type_ii", "type_i", "budget"]:
        save_metric_plot(metrics, key, out_dir)

    print_final_summary(metrics)
    print(f"[plot] wrote cumulative paper-style figures to {out_dir}")


if __name__ == "__main__":
    main()
