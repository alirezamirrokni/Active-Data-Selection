import argparse
import re
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd
import seaborn as sns

from utils import safe_name


PLOT_METRICS = {
    "type_ii": {
        "column": "Cum. Type-II",
        "filename": "type_ii",
        "title": "Type-II error",
    },
    "type_i": {
        "column": "Cum. Type-I",
        "filename": "type_i",
        "title": "Type-I error",
    },
    "budget": {
        "column": "Budget",
        "filename": "budget",
        "title": "Budget",
    },
}


KNOWN_METHOD_PREFIXES = ("llm_select", "ours_llm", "random", "ours")


def _first_nonempty(series: pd.Series) -> str | None:
    for value in series.dropna().tolist():
        text = str(value).strip()
        if text:
            return text
    return None


def _known_dataset_from_text(text: str) -> str | None:
    """Extract dataset names used by this project from a run/cache stem."""
    text = str(text)

    m = re.search(r"(triviaqa500-[^_]+_n[^_]+_seed[^_]+)", text)
    if m:
        return m.group(1)

    m = re.search(r"(math500-[^_]+)", text)
    if m:
        return m.group(1)

    return None


def infer_dataset_name(csv_path: Path, df: pd.DataFrame) -> str:
    """Infer the dataset identifier for a run CSV."""
    if "dataset" in df.columns:
        explicit = _first_nonempty(df["dataset"])
        if explicit:
            return safe_name(explicit)

    stem = csv_path.stem
    if "config" in df.columns:
        config_name = _first_nonempty(df["config"])
        if config_name:
            stem = config_name

    known = _known_dataset_from_text(stem)
    if known:
        return safe_name(known)

    if "_budget" in stem:
        before_budget = stem.split("_budget", 1)[0]

        method = _first_nonempty(df["method"]) if "method" in df.columns else None
        if method:
            method_prefix = str(method).lower().replace("-", "_") + "_"
            if before_budget.startswith(method_prefix):
                before_budget = before_budget[len(method_prefix) :]
        else:
            for prefix in KNOWN_METHOD_PREFIXES:
                method_prefix = prefix + "_"
                if before_budget.startswith(method_prefix):
                    before_budget = before_budget[len(method_prefix) :]
                    break

        main_llm = _first_nonempty(df["main_llm"]) if "main_llm" in df.columns else None
        if main_llm:
            main_prefix = safe_name(main_llm) + "_"
            if before_budget.startswith(main_prefix):
                dataset = before_budget[len(main_prefix) :]
                if dataset:
                    return safe_name(dataset)

        parts = before_budget.split("_", 1)
        if len(parts) == 2 and parts[1]:
            return safe_name(parts[1])

    return safe_name("unknown_dataset")


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
            df["run_path"] = str(csv)
            df["dataset"] = infer_dataset_name(csv, df)
            frames.append(df)

    if not frames:
        raise RuntimeError("No method run CSV files found.")

    return pd.concat(frames, ignore_index=True)


def pretty_method(name: str) -> str:
    mapping = {
        "ours": "Ours",
        "ours_llm": "Ours-LLM",
        "random": "Random",
        "llm_select": "LLM-Select",
    }
    return mapping.get(str(name), str(name))


def safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def summarize_per_round(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "config", "method", "run_file", "t"]

    for (dataset, config, method, run_file, t), g in df.groupby(group_cols, sort=True):
        selected = g["selected"].astype(int)
        A = g["A"].astype(int)
        cost = g["cost"].astype(float)

        n_sel = int(selected.sum())
        n_unsel = int((1 - selected).sum())

        selected_correct = float((selected * (1 - A)).sum())
        unselected_wrong = float(((1 - selected) * A).sum())
        used_budget = float((selected * cost).sum())

        rows.append(
            {
                "dataset": dataset,
                "config": config,
                "method": method,
                "run_file": run_file,
                "Method": pretty_method(method),
                "Round": int(t),
                "Type-I": safe_div(selected_correct, n_sel),
                "Type-II": safe_div(unselected_wrong, n_unsel),
                "Budget": used_budget,
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
        ["dataset", "config", "method", "run_file", "Round"]
    ).reset_index(drop=True)

    group_cols = ["dataset", "config", "method", "run_file"]
    round_index = round_df.groupby(group_cols).cumcount() + 1

    round_df["Cum. Type-I"] = round_df.groupby(group_cols)["Type-I"].cumsum() / round_index
    round_df["Cum. Type-II"] = round_df.groupby(group_cols)["Type-II"].cumsum() / round_index
    round_df["Avg. Budget"] = round_df.groupby(group_cols)["Budget"].cumsum() / round_index

    # Backward-compatible alias for summaries produced by older versions of the plotter.
    # The budget plot itself uses the per-round "Budget" column, not this running average.
    round_df["Cum. Budget"] = round_df["Avg. Budget"]

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

    eps = raw_df[["dataset", "config", "method", "run_file", "epsilon"]].drop_duplicates()
    eps = eps.dropna(subset=["epsilon"])
    if eps.empty:
        metrics["epsilon"] = pd.NA
        return metrics

    return metrics.merge(eps, on=["dataset", "config", "method", "run_file"], how="left")


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


def _pad_ylim_for_reference_line(ax: plt.Axes, y_value: float) -> None:
    """Ensure an annotated horizontal reference line is not clipped."""
    ymin, ymax = ax.get_ylim()
    if ymin <= y_value <= ymax:
        return

    span = ymax - ymin
    if span <= 0:
        span = max(abs(y_value), 1.0)

    pad = 0.06 * span
    ymin = min(ymin, y_value - pad)
    ymax = max(ymax, y_value + pad)
    ax.set_ylim(ymin, ymax)


def _annotate_reference_line(ax: plt.Axes, y_value: float, text: str) -> None:
    """Place a reference-line label just inside the plotting area, near the y-axis."""
    _pad_ylim_for_reference_line(ax, y_value)
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.015,
        y_value,
        text,
        transform=transform,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        clip_on=False,
    )


def _draw_metric_axis(metrics: pd.DataFrame, metric_key: str, ax: plt.Axes, legend: bool) -> None:
    meta = PLOT_METRICS[metric_key]
    y = meta["column"]

    sns.lineplot(
        data=metrics,
        x="Round",
        y=y,
        hue="Method",
        linewidth=2.0,
        errorbar=None,
        legend=legend,
        ax=ax,
    )

    if metric_key == "type_i" and metrics["epsilon"].notna().any():
        epsilon = float(metrics["epsilon"].dropna().iloc[0])
        ax.axhline(
            epsilon,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label="_nolegend_",
        )
        _annotate_reference_line(ax, epsilon, r"$\epsilon$")

    if metric_key == "budget":
        limit = float(metrics["Limit"].iloc[0])
        ax.axhline(
            limit,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label="_nolegend_",
        )
        _annotate_reference_line(ax, limit, "Limit")

    ax.set_xlabel("Round")
    ax.set_ylabel("")
    ax.set_title(meta["title"])
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    sns.despine(ax=ax)


def _dedupe_legend(handles: Iterable[Any], labels: Iterable[str]) -> Tuple[List[Any], List[str]]:
    seen = set()
    out_handles = []
    out_labels = []
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    return out_handles, out_labels


def save_metric_plot(metrics: pd.DataFrame, metric_key: str, out_dir: Path) -> None:
    meta = PLOT_METRICS[metric_key]

    fig, ax = plt.subplots(figsize=(4.35, 2.85))
    _draw_metric_axis(metrics, metric_key, ax=ax, legend=True)

    handles, labels = _dedupe_legend(*ax.get_legend_handles_labels())
    if handles:
        ax.legend(handles, labels, title=None, loc="best")

    fig.tight_layout(pad=0.35)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{meta['filename']}.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_combined_metric_plot(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.25))

    all_handles: List[Any] = []
    all_labels: List[str] = []
    for ax, key in zip(axes, ["type_ii", "type_i", "budget"]):
        _draw_metric_axis(metrics, key, ax=ax, legend=True)
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = _dedupe_legend(all_handles, all_labels)
    if handles:
        ncol = min(max(len(labels), 1), 6)
        fig.legend(
            handles,
            labels,
            title=None,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            frameon=False,
        )

    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), pad=0.45, w_pad=1.8)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"combined_metrics.{ext}", bbox_inches="tight")
    plt.close(fig)


def print_final_summary(metrics: pd.DataFrame, dataset: str) -> None:
    final_rows = (
        metrics.sort_values(["dataset", "config", "method", "run_file", "Round"])
        .groupby(["dataset", "config", "method", "run_file"], as_index=False)
        .tail(1)
    )

    cols = [
        "Method",
        "Round",
        "Cum. Type-I",
        "Cum. Type-II",
        "Pooled Type-I",
        "Pooled Type-II",
        "Avg. Budget",
        "Limit",
    ]

    print()
    print(f"[plot] final cumulative summary for {dataset}")
    print(final_rows[cols].to_string(index=False))


def write_dataset_plots(raw_df: pd.DataFrame, root_out_dir: Path, dataset: str) -> None:
    out_dir = root_out_dir / safe_name(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    save_combined_metric_plot(metrics, out_dir)

    print_final_summary(metrics, dataset)
    print(f"[plot] wrote figures to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs"],
        help="Run CSVs or output directories. Directories are scanned for method-run CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        default="figures",
        help="Root directory for plots. A subdirectory is created for each dataset.",
    )
    args = parser.parse_args()

    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    raw_df = collect_run_csvs(args.runs)

    datasets = sorted(raw_df["dataset"].dropna().unique().tolist())
    if not datasets:
        raise RuntimeError("No datasets could be inferred from the run CSV files.")

    print("[plot] discovered datasets: " + ", ".join(datasets))
    for dataset in datasets:
        dataset_df = raw_df[raw_df["dataset"] == dataset].copy()
        write_dataset_plots(dataset_df, root_out_dir, dataset)


if __name__ == "__main__":
    main()
