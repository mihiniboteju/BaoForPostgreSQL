#!/usr/bin/env python3
"""
analyze_bao_vs_pg.py

Compares two query-execution logs (Bao-enabled vs plain PostgreSQL) produced
by the instrumented run_queries.py described earlier, and produces a set of
plots + a text summary.

Expected CSV columns (header row required):
    query_id, t_start, t_end, duration_sec, elapsed_since_t0_sec, run_type

Usage:
    python3 analyze_bao_vs_pg.py --pg pg_baseline_log.csv --bao bao_log.csv --outdir plots/ --retrain-window 25

All plots are written as PNG files to --outdir. A text summary is printed
to stdout and also written to <outdir>/summary.txt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_log(path: str, run_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Derive duration_sec if not present (e.g. logs stored duration_ms instead)
    if "duration_sec" not in df.columns:
        if "duration_ms" in df.columns:
            df["duration_sec"] = df["duration_ms"] / 1000.0
        elif {"t_start", "t_end"}.issubset(df.columns):
            df["duration_sec"] = df["t_end"] - df["t_start"]
        else:
            sys.exit(f"ERROR: {path} has no duration_sec, duration_ms, or "
                      f"t_start/t_end pair to derive duration from.")

    # Derive elapsed_since_t0_sec if not present: time since this run's
    # first query started, using t_end (completion time) as the reference
    # point for each row, and this run's earliest t_start as t0.
    if "elapsed_since_t0_sec" not in df.columns:
        if "t_end" not in df.columns or "t_start" not in df.columns:
            sys.exit(f"ERROR: {path} has no elapsed_since_t0_sec and no "
                      f"t_start/t_end to derive it from.")
        t0 = df["t_start"].min()
        df["elapsed_since_t0_sec"] = df["t_end"] - t0

    required = {"query_id", "t_start", "t_end", "duration_sec", "elapsed_since_t0_sec"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required columns: {missing}")

    df = df.sort_values("elapsed_since_t0_sec").reset_index(drop=True)
    df["run_type"] = run_type
    df["query_index"] = np.arange(1, len(df) + 1)          # completion order
    df["cumulative_count"] = df["query_index"]              # alias, same thing

    # Inter-query gap: time between the previous query's completion and this
    # query's start. In a purely sequential, back-to-back run this should be
    # ~0. A large positive gap means something (e.g. Bao's periodic model
    # retraining) blocked the pipeline between queries, and that dead time
    # is invisible to duration_sec but very real in wall-clock terms.
    prev_t_end = df["t_end"].shift(1)
    df["gap_sec"] = (df["t_start"] - prev_t_end).clip(lower=0)
    df.loc[0, "gap_sec"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_crossover(pg: pd.DataFrame, bao: pd.DataFrame):
    """
    Find the wall-clock time after which Bao's cumulative completed-query
    count SUSTAINABLY exceeds PostgreSQL's (i.e. stays ahead for the rest of
    the overlapping window), not just a momentary/noisy lead early on.

    This deliberately ignores early, transient crossings (e.g. Bao's very
    first query happening to finish slightly before PG's first query, purely
    by chance) and instead reports the point after which Bao is continuously
    ahead through the end of the overlapping time range. Uses step-function
    (previous-value-hold) interpolation, since both logs represent strictly
    sequential, one-at-a-time completions.

    Returns (crossover_time_sec, pg_count_at_crossover, bao_count_at_crossover)
    or None if Bao never ends up sustainably ahead within the overlap window.
    """
    t_max = min(pg["elapsed_since_t0_sec"].max(), bao["elapsed_since_t0_sec"].max())
    if t_max <= 0:
        return None

    grid = np.linspace(0, t_max, 20000)

    def step_lookup(df):
        # previous-value-hold: count of completed queries at or before time t
        return np.searchsorted(df["elapsed_since_t0_sec"].values, grid, side="right")

    pg_counts = step_lookup(pg)
    bao_counts = step_lookup(bao)

    diff = bao_counts - pg_counts
    positive_mask = diff > 0
    if not positive_mask.any():
        return None

    nonpositive_idx = np.where(~positive_mask)[0]
    if len(nonpositive_idx) == 0:
        # Bao was ahead from t=0 onward (unusual, but handle it)
        i = 0
    else:
        i = nonpositive_idx[-1] + 1
        if i >= len(grid):
            return None

    return grid[i], int(pg_counts[i]), int(bao_counts[i])


def summarize(pg: pd.DataFrame, bao: pd.DataFrame, crossover) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)

    for name, df in [("PostgreSQL (baseline)", pg), ("Bao", bao)]:
        total_time = df["elapsed_since_t0_sec"].max()
        lines.append(f"\n{name}:")
        lines.append(f"  queries completed        : {len(df)}")
        lines.append(f"  total wall-clock time     : {total_time:,.1f} sec  ({total_time/3600:.2f} hr)")
        lines.append(f"  mean duration             : {df['duration_sec'].mean():.3f} sec")
        lines.append(f"  median (p50) duration     : {df['duration_sec'].median():.3f} sec")
        lines.append(f"  p90 duration              : {df['duration_sec'].quantile(0.90):.3f} sec")
        lines.append(f"  p95 duration              : {df['duration_sec'].quantile(0.95):.3f} sec")
        lines.append(f"  p99 duration              : {df['duration_sec'].quantile(0.99):.3f} sec")
        lines.append(f"  max duration              : {df['duration_sec'].max():.3f} sec")
        total_gap = df["gap_sec"].sum()
        lines.append(f"  sum of logged durations   : {df['duration_sec'].sum():,.1f} sec")
        lines.append(f"  total inter-query gap time: {total_gap:,.1f} sec "
                      f"({100 * total_gap / max(total_time, 1e-9):.1f}% of wall-clock time)")

    # Throughput at matched wall-clock time (fair comparison point)
    t_max = min(pg["elapsed_since_t0_sec"].max(), bao["elapsed_since_t0_sec"].max())
    pg_at_tmax = (pg["elapsed_since_t0_sec"] <= t_max).sum()
    bao_at_tmax = (bao["elapsed_since_t0_sec"] <= t_max).sum()
    lines.append(f"\nAt matched elapsed time ({t_max:,.1f} sec, the shorter run's total duration):")
    lines.append(f"  PostgreSQL completed      : {pg_at_tmax} queries")
    lines.append(f"  Bao completed             : {bao_at_tmax} queries")
    lines.append(f"  Bao / PG ratio            : {bao_at_tmax / max(pg_at_tmax, 1):.3f}")

    if crossover is not None:
        ct, pgc, baoc = crossover
        lines.append(f"\nCrossover point (Bao cumulative count first exceeds PG's):")
        lines.append(f"  elapsed time              : {ct:,.1f} sec  ({ct/3600:.2f} hr)")
        lines.append(f"  PG count at that time     : {pgc}")
        lines.append(f"  Bao count at that time    : {baoc}")
    else:
        lines.append(f"\nCrossover point: NOT REACHED within the overlapping time window "
                      f"(0 to {t_max:,.1f} sec). Bao never completed more cumulative queries "
                      f"than PostgreSQL in this run.")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cumulative_curve(pg, bao, crossover, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(pg["elapsed_since_t0_sec"] / 60, pg["cumulative_count"],
            where="post", label="PostgreSQL (baseline)", color="tab:blue", linewidth=1.8)
    ax.step(bao["elapsed_since_t0_sec"] / 60, bao["cumulative_count"],
            where="post", label="Bao", color="tab:orange", linewidth=1.8)

    if crossover is not None:
        ct, pgc, baoc = crossover
        ax.axvline(ct / 60, color="gray", linestyle="--", linewidth=1)
        ax.scatter([ct / 60], [baoc], color="tab:orange", zorder=5, s=50)
        ax.annotate(f"crossover\n~{ct/60:.1f} min",
                    xy=(ct / 60, baoc), xytext=(10, -30),
                    textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Elapsed wall-clock time (minutes)")
    ax.set_ylabel("Cumulative queries completed")
    ax.set_title("Cumulative Queries Completed vs. Elapsed Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "01_cumulative_curve.png", dpi=150)
    plt.close(fig)


def plot_duration_cumulative_curve(pg, bao, outdir: Path):
    """Compare query completion using durations only, with no inter-query gaps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for df, label, color in [(pg, "PostgreSQL (baseline)", "tab:blue"),
                             (bao, "Bao", "tab:orange")]:
        elapsed_minutes = df["duration_sec"].cumsum() / 60
        ax.step(elapsed_minutes, df["cumulative_count"], where="post",
                label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Cumulative query time (minutes, no gaps)")
    ax.set_ylabel("Cumulative queries completed")
    ax.set_title("Cumulative Queries Completed Using Query Durations Only")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "07_duration_cumulative_curve.png", dpi=150)
    plt.close(fig)


def plot_rolling_latency(pg, bao, retrain_window: int, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for df, label, color in [(pg, "PostgreSQL (baseline)", "tab:blue"),
                              (bao, "Bao", "tab:orange")]:
        roll = df["duration_sec"].rolling(window=retrain_window, min_periods=1).mean()
        ax.plot(df["query_index"], roll, label=f"{label} (rolling mean, window={retrain_window})",
                color=color, linewidth=1.8)

    # Mark Bao retrain checkpoints, but cap how many lines we draw so the
    # plot stays readable even with thousands of queries / a small window.
    n_bao = len(bao)
    all_checkpoints = np.arange(retrain_window, n_bao, retrain_window)
    max_lines = 60
    step = max(1, len(all_checkpoints) // max_lines)
    checkpoints = all_checkpoints[::step]
    for i, cp in enumerate(checkpoints):
        ax.axvline(cp, color="gray", linestyle=":", linewidth=0.7, alpha=0.6,
                   label="Bao retrain checkpoint" if i == 0 else None)

    ax.set_xlabel("Query index (completion order)")
    ax.set_ylabel(f"Rolling mean duration (sec, window={retrain_window})")
    ax.set_title("Rolling Average Query Latency — Cold Start / Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "02_rolling_latency.png", dpi=150)
    plt.close(fig)


def plot_latency_distribution(pg, bao, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ECDF
    ax = axes[0]
    for df, label, color in [(pg, "PostgreSQL", "tab:blue"), (bao, "Bao", "tab:orange")]:
        x = np.sort(df["duration_sec"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=label, color=color, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Query duration (sec, log scale)")
    ax.set_ylabel("Cumulative fraction of queries")
    ax.set_title("Latency ECDF")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Percentile bar chart
    ax = axes[1]
    percentiles = [50, 90, 95, 99]
    pg_vals = [pg["duration_sec"].quantile(p / 100) for p in percentiles]
    bao_vals = [bao["duration_sec"].quantile(p / 100) for p in percentiles]
    x = np.arange(len(percentiles))
    width = 0.35
    ax.bar(x - width / 2, pg_vals, width, label="PostgreSQL", color="tab:blue")
    ax.bar(x + width / 2, bao_vals, width, label="Bao", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{p}" for p in percentiles])
    ax.set_ylabel("Duration (sec)")
    ax.set_title("Latency Percentiles")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outdir / "03_latency_distribution.png", dpi=150)
    plt.close(fig)


def plot_raw_scatter(pg, bao, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pg["query_index"], pg["duration_sec"], s=6, alpha=0.4,
               label="PostgreSQL", color="tab:blue")
    ax.scatter(bao["query_index"], bao["duration_sec"], s=6, alpha=0.4,
               label="Bao", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("Query index (completion order)")
    ax.set_ylabel("Duration (sec, log scale)")
    ax.set_title("Per-Query Latency (raw) — outliers and variance")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outdir / "04_raw_scatter.png", dpi=150)
    plt.close(fig)


def plot_interquery_gaps(pg, bao, retrain_window: int, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(pg["query_index"], pg["gap_sec"], s=8, alpha=0.5,
               label="PostgreSQL", color="tab:blue")
    ax.scatter(bao["query_index"], bao["gap_sec"], s=8, alpha=0.5,
               label="Bao", color="tab:orange")

    # mark retrain checkpoints (thinned, same logic as the rolling-latency plot)
    n_bao = len(bao)
    all_checkpoints = np.arange(retrain_window, n_bao, retrain_window)
    max_lines = 60
    step = max(1, len(all_checkpoints) // max_lines)
    checkpoints = all_checkpoints[::step]
    for i, cp in enumerate(checkpoints):
        ax.axvline(cp, color="gray", linestyle=":", linewidth=0.7, alpha=0.5,
                   label="Bao retrain checkpoint" if i == 0 else None)

    ax.set_xlabel("Query index (completion order)")
    ax.set_ylabel("Gap before this query started (sec)")
    ax.set_title("Inter-Query Dead Time — reveals pipeline stalls (e.g. retraining)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "05_interquery_gaps.png", dpi=150)
    plt.close(fig)


def plot_gap_phase_analysis(bao: pd.DataFrame, retrain_window: int, outdir: Path):
    """
    Plots gap_sec against (query_index mod retrain_window) instead of raw
    query_index. If large gaps are caused by periodic retraining, they should
    cluster tightly at a specific phase (typically right after a checkpoint),
    regardless of how spread out they look along the raw query-index axis.
    This is a more rigorous check than eyeballing gridline alignment, since
    it doesn't depend on how many checkpoint lines happen to get drawn.
    """
    phase = bao["query_index"] % retrain_window

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(phase, bao["gap_sec"], s=10, alpha=0.5, color="tab:orange")
    ax.set_xlabel(f"Query index mod {retrain_window}  (0 = right after a retrain checkpoint)")
    ax.set_ylabel("Gap before this query started (sec)")
    ax.set_title("Gap Phase Analysis — do large gaps cluster right after a checkpoint?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "06_gap_phase_analysis.png", dpi=150)
    plt.close(fig)

    # Numeric summary: is mean gap size near phase 0 meaningfully higher
    # than mean gap size elsewhere? A simple, threshold-free check.
    near_checkpoint = phase <= max(2, retrain_window // 20)   # first ~5% of window
    mean_near = bao.loc[near_checkpoint, "gap_sec"].mean()
    mean_far = bao.loc[~near_checkpoint, "gap_sec"].mean()

    # Where do the biggest gaps actually fall, phase-wise?
    top_gaps = bao.nlargest(15, "gap_sec")[["query_index", "gap_sec"]].copy()
    top_gaps["phase"] = top_gaps["query_index"] % retrain_window

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("GAP PHASE ANALYSIS (is dead time tied to retrain checkpoints?)")
    lines.append("=" * 70)
    lines.append(f"  mean gap within {near_checkpoint.sum()} queries of a checkpoint : {mean_near:.2f} sec")
    lines.append(f"  mean gap elsewhere                                  : {mean_far:.2f} sec")
    lines.append(f"\n  Top 15 largest gaps (query_index, gap_sec, phase = index mod {retrain_window}):")
    for _, row in top_gaps.iterrows():
        lines.append(f"    idx={int(row['query_index']):>5}   gap={row['gap_sec']:>7.2f}s   phase={int(row['phase']):>4}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pg", required=True, help="Path to plain-PostgreSQL log CSV")
    ap.add_argument("--bao", required=True, help="Path to Bao log CSV")
    ap.add_argument("--outdir", default="plots", help="Directory to write plots + summary into")
    ap.add_argument("--retrain-window", type=int, default=25,
                     help="Bao's retrain window size, for rolling-average annotations")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pg = load_log(args.pg, "PG")
    bao = load_log(args.bao, "BAO")

    crossover = find_crossover(pg, bao)

    plot_cumulative_curve(pg, bao, crossover, outdir)
    plot_duration_cumulative_curve(pg, bao, outdir)
    plot_rolling_latency(pg, bao, args.retrain_window, outdir)
    plot_latency_distribution(pg, bao, outdir)
    plot_raw_scatter(pg, bao, outdir)
    plot_interquery_gaps(pg, bao, args.retrain_window, outdir)
    gap_phase_summary = plot_gap_phase_analysis(bao, args.retrain_window, outdir)

    summary = summarize(pg, bao, crossover)
    summary += "\n" + gap_phase_summary
    print(summary)
    (outdir / "summary.txt").write_text(summary)

    print(f"\nPlots written to: {outdir.resolve()}")
    print(" - 01_cumulative_curve.png   (the main 'does Bao overtake PG' plot)")
    print(" - 02_rolling_latency.png    (cold start / learning-curve visualization)")
    print(" - 03_latency_distribution.png (ECDF + percentile comparison)")
    print(" - 04_raw_scatter.png        (raw per-query variance/outliers)")
    print(" - 05_interquery_gaps.png    (dead-time between queries — retrain stalls etc.)")
    print(" - 06_gap_phase_analysis.png (does dead time cluster right after retrain checkpoints?)")
    print(" - 07_duration_cumulative_curve.png (query durations only, no gaps)")


if __name__ == "__main__":
    main()