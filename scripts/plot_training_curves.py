#!/usr/bin/env python3
# scripts/plot_training_curves.py
#
# Reads data/eval_log.jsonl and generates training curve plots for the paper.
#
# Usage:
#   uv run python scripts/plot_training_curves.py
#   uv run python scripts/plot_training_curves.py --out data/artifacts/plots/
#
# Produces four figures:
#   1. mean_score.png        — mean final score per iteration (± std band)
#   2. bid_error.png         — mean |bid − won| per iteration
#   3. grimoire_size.png     — number of rules per iteration
#   4. mean_fitness.png      — mean rule fitness per iteration

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

EVAL_LOG = Path(__file__).resolve().parent.parent / "data" / "eval_log.jsonl"

PERSONA_COLOURS = {
    "rational":    "#2196F3",   # blue
    "forced_zero": "#FF5722",   # deep orange
    "heuristic":   "#9E9E9E",   # grey
}
PERSONA_LABELS = {
    "rational":    "Rational",
    "forced_zero": "Forced Zero",
    "heuristic":   "Heuristic",
}


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Eval log not found: {path}\n"
            "Run the training loop first: ./run_loop.sh"
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["iteration"])
    return records


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Mean score per iteration
# ─────────────────────────────────────────────────────────────────────────────

def plot_score(records, out_dir: Path):
    # Discover all personas present across records
    personas = sorted({
        p
        for r in records
        for p in r.get("by_persona", {})
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for persona in personas:
        iters  = []
        means  = []
        stds   = []
        for r in records:
            stats = r.get("by_persona", {}).get(persona, {}).get("score", {})
            if stats.get("mean") is not None:
                iters.append(r["iteration"])
                means.append(stats["mean"])
                stds.append(stats.get("std", 0.0))

        if not iters:
            continue

        colour = PERSONA_COLOURS.get(persona, "#607D8B")
        label  = PERSONA_LABELS.get(persona, persona)
        ax.plot(iters, means, marker="o", markersize=5, color=colour, label=label, linewidth=2)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(iters, lo, hi, color=colour, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    _style(ax, "Mean Final Score per Iteration", "Iteration", "Mean Score")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "mean_score.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Mean bid error per iteration
# ─────────────────────────────────────────────────────────────────────────────

def plot_bid_error(records, out_dir: Path):
    personas = sorted({
        p
        for r in records
        for p in r.get("by_persona", {})
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for persona in personas:
        iters = []
        means = []
        stds  = []
        for r in records:
            stats = r.get("by_persona", {}).get(persona, {}).get("bid_error", {})
            if stats.get("mean") is not None:
                iters.append(r["iteration"])
                means.append(stats["mean"])
                stds.append(stats.get("std", 0.0))

        if not iters:
            continue

        colour = PERSONA_COLOURS.get(persona, "#607D8B")
        label  = PERSONA_LABELS.get(persona, persona)
        ax.plot(iters, means, marker="o", markersize=5, color=colour, label=label, linewidth=2)
        lo = [max(0, m - s) for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(iters, lo, hi, color=colour, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    _style(ax, "Mean Bid Error per Iteration  (|bid − won|, lower is better)",
           "Iteration", "Mean |bid − won|")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "bid_error.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Grimoire size per iteration
# ─────────────────────────────────────────────────────────────────────────────

def plot_grimoire_size(records, out_dir: Path):
    persona_keys = sorted({
        p
        for r in records
        for p in r.get("grimoire", {})
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for persona in persona_keys:
        iters = []
        sizes = []
        for r in records:
            g = r.get("grimoire", {}).get(persona, {})
            if g.get("size") is not None:
                iters.append(r["iteration"])
                sizes.append(g["size"])

        if not iters:
            continue

        colour = PERSONA_COLOURS.get(persona, "#607D8B")
        label  = PERSONA_LABELS.get(persona, persona)
        ax.plot(iters, sizes, marker="s", markersize=5, color=colour, label=label, linewidth=2)

    _style(ax, "Grimoire Size per Iteration", "Iteration", "Number of Rules")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "grimoire_size.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Mean rule fitness per iteration
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_fitness(records, out_dir: Path):
    persona_keys = sorted({
        p
        for r in records
        for p in r.get("grimoire", {})
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for persona in persona_keys:
        iters    = []
        fitnesses = []
        for r in records:
            g = r.get("grimoire", {}).get(persona, {})
            if g.get("mean_fitness") is not None:
                iters.append(r["iteration"])
                fitnesses.append(g["mean_fitness"])

        if not iters:
            continue

        colour = PERSONA_COLOURS.get(persona, "#607D8B")
        label  = PERSONA_LABELS.get(persona, persona)
        ax.plot(iters, fitnesses, marker="^", markersize=5, color=colour, label=label, linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    _style(ax, "Mean Rule Fitness per Iteration", "Iteration", "Mean Fitness")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "mean_fitness.png")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: Zero bid success rate (forced_zero only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_zero_success(records, out_dir: Path):
    iters   = []
    rates   = []
    for r in records:
        zsr = r.get("by_persona", {}).get("forced_zero", {}).get("zero_success_rate")
        if zsr is not None:
            iters.append(r["iteration"])
            rates.append(zsr * 100)   # convert to %

    if not iters:
        return   # no forced_zero data

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colour = PERSONA_COLOURS["forced_zero"]
    ax.plot(iters, rates, marker="D", markersize=5, color=colour, linewidth=2, label="Forced Zero")
    ax.axhline(100, color="black", linewidth=0.8, linestyle=":", label="Perfect (100%)")
    _style(ax, "Zero-Bid Success Rate per Iteration  (Forced Zero)",
           "Iteration", "Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    _save(fig, out_dir, "zero_success_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: ELO ratings per iteration
# ─────────────────────────────────────────────────────────────────────────────

def plot_elo(records, out_dir: Path):
    personas = sorted({
        p
        for r in records
        for p in r.get("elo", {})
    })

    if not personas:
        return   # no ELO data

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for persona in personas:
        iters  = []
        ratings = []
        for r in records:
            elo = r.get("elo", {}).get(persona)
            if elo is not None:
                iters.append(r["iteration"])
                ratings.append(elo)

        if not iters:
            continue

        colour = PERSONA_COLOURS.get(persona, "#607D8B")
        label  = PERSONA_LABELS.get(persona, persona)
        ax.plot(iters, ratings, marker="o", markersize=5, color=colour, label=label, linewidth=2)

    ax.axhline(1000, color="black", linewidth=0.8, linestyle=":", label="Baseline (1000)")
    _style(ax, "ELO Rating per Iteration", "Iteration", "ELO Rating")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "elo_ratings.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text summary (fallback if matplotlib not installed)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records):
    print(f"\n{'─' * 70}")
    print(f"  Evaluation log — {len(records)} checkpoint(s)")
    print(f"{'─' * 70}")
    for r in records:
        print(f"\n  Iteration {r['iteration']:>3d}  ({r['timestamp']})")
        for persona, stats in r.get("by_persona", {}).items():
            score = stats.get("score", {})
            berr  = stats.get("bid_error", {})
            line  = (
                f"    {persona:15s}  "
                f"score={score.get('mean', 'N/A'):>+7.1f} ± {score.get('std', 0):>5.1f}  "
                f"bid_err={berr.get('mean', 'N/A'):>5.3f}"
            )
            if "zero_success_rate" in stats and stats["zero_success_rate"] is not None:
                line += f"  zero_ok={stats['zero_success_rate']:.1%}"
            print(line)
        for persona, g in r.get("grimoire", {}).items():
            print(f"    Grimoire [{persona:12s}]  rules={g['size']}  "
                  f"mean_fitness={g['mean_fitness']}")
        elo = r.get("elo", {})
        if elo:
            elo_str = "  ".join(f"{p}={v}" for p, v in sorted(elo.items()))
            print(f"    ELO: {elo_str}")
    print(f"{'─' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot SkullKingAI training curves")
    parser.add_argument(
        "--log", default=str(EVAL_LOG),
        help="Path to eval_log.jsonl (default: data/eval_log.jsonl)"
    )
    parser.add_argument(
        "--out", default="data/artifacts/plots",
        help="Output directory for PNG files (default: data/artifacts/plots)"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir  = Path(args.out)

    try:
        records = _load(log_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loaded {len(records)} evaluation checkpoint(s) from {log_path}")

    # Always print the text summary
    print_summary(records)

    if not HAS_MPL:
        print("matplotlib not installed — skipping plot generation.")
        print("Install with:  uv add matplotlib")
        return

    if len(records) < 2:
        print("Need at least 2 checkpoints to plot curves. Run more iterations.")
        return

    print(f"Generating plots → {out_dir}/")
    plot_score(records, out_dir)
    plot_bid_error(records, out_dir)
    plot_grimoire_size(records, out_dir)
    plot_mean_fitness(records, out_dir)
    plot_zero_success(records, out_dir)
    plot_elo(records, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
