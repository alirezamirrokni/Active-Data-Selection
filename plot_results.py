import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLOT_METRICS = {
    "type_ii": {"column": "Type-II", "ylabel": "Type-II", "filename": "type_ii"},
    "type_i": {"column": "Type-I", "ylabel": "Type-I", "filename": "type_i"},
    "budget": {"column": "Budget", "ylabel": "Budget", "filename": "budget"},
}


def collect_run_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            csvs = sorted(path.glob("*.csv"))
        else:
            csvs = [path]
        for csv in csvs:
            if csv.name.startswith("gen_") or csv.name.endswith("_metrics.csv"):
                continue
            frames.append(pd.read_csv(csv))
    if not frames:
        raise RuntimeError("No method run CSV files found.")
    return pd.concat(frames, ignore_index=True)


def pretty_method(name: str) -> str:
    mapping = {
        "ours": "Ours",
        "random": "Random",
    }
    return mapping.get(str(name), str(name))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, method, t), g in df.groupby(["config", "method", "t"], sort=True):
        selected = g["selected"].astype(int)
        A = g["A"].astype(int)
        cost = g["cost"].astype(float)
        n_sel = int(selected.sum())
        n_unsel = int((1 - selected).sum())
        type_i = 0.0 if n_sel == 0 else float((selected * (1 - A)).sum() / n_sel)
        type_ii = 0.0 if n_unsel == 0 else float(((1 - selected) * A).sum() / n_unsel)
        rows.append(
            {
                "config": config,
                "method": method,
                "Method": pretty_method(method),
                "Round": int(t),
                "Type-I": type_i,
                "Type-II": type_ii,
                "Budget": float((selected * cost).sum()),
                "Limit": float(g["budget"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def set_paper_style():
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
        marker="o",
        linewidth=2.0,
        markersize=4.2,
        ax=ax,
    )

    if metric_key == "budget":
        limit = float(metrics["Limit"].iloc[0])
        ax.axhline(limit, linestyle="--", linewidth=1.3, color="black", label="Limit")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["outputs"], help="Run CSVs or output directories.")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    df = collect_run_csvs(args.runs)
    metrics = summarize(df)
    metrics_path = out_dir / "summary_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[plot] wrote {metrics_path}")

    for key in ["type_ii", "type_i", "budget"]:
        save_metric_plot(metrics, key, out_dir)
    print(f"[plot] wrote paper-style figures to {out_dir}")


if __name__ == "__main__":
    main()
