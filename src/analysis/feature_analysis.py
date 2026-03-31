"""
Feature count vs accuracy analysis.

Correlates dialect feature density with accuracy gap at multiple thresholds
to address second reader feedback about min_features=1 being too low.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .presentation import (
    ACCURACY_BENCHMARKS,
    AUDIT_COMBINED_RE,
    BENCHMARK_LABELS,
    DIALECT_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    TOXICITY_BENCHMARKS,
    _setup_style,
    _save,
    format_benchmark,
    format_dialect,
    AUDIT_FILENAME_RE,
)

DIALECT_COLORS = {
    "aave": "#E64B35",
    "hiberno_english": "#4DBBD5",
    "indian_english": "#00A087",
}


def _find_generation_file(benchmark: str, dialect: str, bench_dir: Path) -> Path | None:
    """Find the generation file for a benchmark-dialect pair."""
    # Map audit benchmark names to generation file prefixes
    bench_prefix_map = {
        "arc_challenge": "arc",
        "realtoxicityprompts": "realtoxicityprompts",
    }
    prefix = bench_prefix_map.get(benchmark, benchmark)

    for f in bench_dir.glob(f"{prefix}_{dialect}_*.json"):
        return f
    return None


def load_feature_accuracy_data(
    audit_base: Path,
    bench_dir: Path,
) -> list[dict]:
    """
    Join audit results with generation data to get per-pair feature counts
    and correctness.

    Returns list of dicts with keys:
        model, benchmark, dialect, pair_id, feature_count,
        original_correct, transformed_correct
    """
    model_dirs = {
        "qwen2.5_14b", "qwen2.5_7b", "llama3.1_8b", "gemma2_9b", "mistral_7b",
    }

    # Pre-load generation data: (benchmark, dialect) -> {id: feature_count}
    gen_cache: dict[tuple[str, str], dict[str, int]] = {}

    records = []
    for d in sorted(audit_base.iterdir()):
        if not d.is_dir() or d.name not in model_dirs:
            continue

        # Deduplicate: prefer combined files over timestamped
        best: dict[tuple[str, str], Path] = {}
        for audit_path in sorted(d.glob("audit_*.json")):
            mc = AUDIT_COMBINED_RE.match(audit_path.name)
            if mc:
                benchmark, dialect, _ = mc.groups()
                best[(benchmark, dialect)] = audit_path
                continue
            m = AUDIT_FILENAME_RE.match(audit_path.name)
            if not m:
                continue
            benchmark, dialect, _, date, time = m.groups()
            key = (benchmark, dialect)
            if key not in best:
                best[key] = audit_path

        for (benchmark, dialect), audit_path in sorted(best.items()):
            # Load or cache generation data
            cache_key = (benchmark, dialect)
            if cache_key not in gen_cache:
                gen_file = _find_generation_file(benchmark, dialect, bench_dir)
                if gen_file is None:
                    gen_cache[cache_key] = {}
                else:
                    with open(gen_file) as f:
                        gen_data = json.load(f)
                    gen_cache[cache_key] = {
                        p["id"]: p.get("feature_count", 1)
                        for p in gen_data["pairs"]
                    }

            fc_lookup = gen_cache[cache_key]

            with open(audit_path) as f:
                audit_data = json.load(f)

            model_name = audit_data["model"]

            for pair in audit_data["pairs"]:
                pid = pair["id"]
                fc = fc_lookup.get(pid)
                if fc is None:
                    continue

                records.append({
                    "model": model_name,
                    "benchmark": benchmark,
                    "dialect": dialect,
                    "pair_id": pid,
                    "feature_count": fc,
                    "original_correct": pair.get("original_correct"),
                    "transformed_correct": pair.get("transformed_correct"),
                })

    return records


def compute_threshold_stats(
    records: list[dict],
    benchmarks: list[str] | None = None,
) -> dict[int, dict]:
    """
    Compute accuracy gap stats at each minimum feature threshold.

    Returns: {threshold: {"n": int, "orig_acc": float, "trans_acc": float, "gap": float}}
    """
    if benchmarks is None:
        benchmarks = ACCURACY_BENCHMARKS

    filtered = [r for r in records if r["benchmark"] in benchmarks]

    max_fc = max(r["feature_count"] for r in filtered) if filtered else 1
    stats = {}
    for threshold in range(1, max_fc + 1):
        subset = [r for r in filtered if r["feature_count"] >= threshold]
        if not subset:
            break
        n = len(subset)
        orig_correct = sum(1 for r in subset if r["original_correct"])
        trans_correct = sum(1 for r in subset if r["transformed_correct"])
        orig_acc = orig_correct / n
        trans_acc = trans_correct / n
        stats[threshold] = {
            "n": n,
            "orig_acc": orig_acc,
            "trans_acc": trans_acc,
            "gap": (trans_acc - orig_acc) * 100,
        }
    return stats


def compute_range_stats(
    records: list[dict],
    benchmarks: list[str] | None = None,
) -> dict[str, dict]:
    """
    Compute accuracy gap for bounded feature count ranges.

    Returns: {"1": {...}, "2": {...}, "3": {...}, "4+": {...}}
    """
    if benchmarks is None:
        benchmarks = ACCURACY_BENCHMARKS

    filtered = [r for r in records if r["benchmark"] in benchmarks]

    ranges = {
        "1": lambda fc: fc == 1,
        "2": lambda fc: fc == 2,
        "3": lambda fc: fc == 3,
        "4+": lambda fc: fc >= 4,
    }

    stats = {}
    for label, pred in ranges.items():
        subset = [r for r in filtered if pred(r["feature_count"])]
        if not subset:
            stats[label] = {"n": 0, "orig_acc": 0, "trans_acc": 0, "gap": 0}
            continue
        n = len(subset)
        orig_correct = sum(1 for r in subset if r["original_correct"])
        trans_correct = sum(1 for r in subset if r["transformed_correct"])
        orig_acc = orig_correct / n
        trans_acc = trans_correct / n
        stats[label] = {
            "n": n,
            "orig_acc": orig_acc,
            "trans_acc": trans_acc,
            "gap": (trans_acc - orig_acc) * 100,
        }
    return stats


def compute_per_model_range_stats(
    records: list[dict],
    benchmarks: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Per-model breakdown of feature range stats."""
    if benchmarks is None:
        benchmarks = ACCURACY_BENCHMARKS

    filtered = [r for r in records if r["benchmark"] in benchmarks]
    models = sorted(set(r["model"] for r in filtered))

    result = {}
    for model in models:
        model_records = [r for r in filtered if r["model"] == model]
        result[model] = compute_range_stats(model_records, benchmarks)
    return result


def chart_feature_threshold(records: list[dict], output_path: Path):
    """Line chart: accuracy gap vs minimum feature threshold, all models pooled."""
    _setup_style()

    stats = compute_threshold_stats(records)
    # Cap at threshold 4 (n=5 has only ~10 pairs, not meaningful)
    thresholds = [t for t in sorted(stats.keys()) if t <= 4]
    gaps = [stats[t]["gap"] for t in thresholds]
    counts = [stats[t]["n"] for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_gap = "#C0392B"
    color_n = "#7F8C8D"

    ax1.bar(thresholds, gaps, color=color_gap, alpha=0.7, width=0.6, label="Accuracy Gap")
    ax1.set_xlabel("Minimum Feature Count")
    ax1.set_ylabel("Accuracy Gap (pp)", color=color_gap)
    ax1.tick_params(axis="y", labelcolor=color_gap)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(thresholds)

    # Annotate gap values
    for t, g in zip(thresholds, gaps):
        va = "top" if g < 0 else "bottom"
        offset = -0.3 if g < 0 else 0.3
        ax1.text(t, g + offset, f"{g:+.1f}", ha="center", va=va, fontsize=10, color=color_gap)

    # Second y-axis for sample count
    ax2 = ax1.twinx()
    ax2.plot(thresholds, counts, "o--", color=color_n, markersize=6, label="Sample Count")
    ax2.set_ylabel("Number of Pairs", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)
    for t, n in zip(thresholds, counts):
        ax2.text(t + 0.15, n, f"n={n}", fontsize=8, color=color_n, va="bottom")

    ax1.set_title("Accuracy Gap by Minimum Feature Count Threshold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    fig.tight_layout()
    _save(fig, output_path)


def chart_feature_range_bars(records: list[dict], output_path: Path):
    """Grouped bar chart: SAE vs dialect accuracy for each feature range."""
    _setup_style()

    stats = compute_range_stats(records)
    ranges = ["1", "2", "3", "4+"]
    ranges = [r for r in ranges if stats.get(r, {}).get("n", 0) > 0]

    orig_accs = [stats[r]["orig_acc"] * 100 for r in ranges]
    trans_accs = [stats[r]["trans_acc"] * 100 for r in ranges]
    gaps = [stats[r]["gap"] for r in ranges]
    counts = [stats[r]["n"] for r in ranges]

    x = np.arange(len(ranges))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars1 = ax.bar(x - width / 2, orig_accs, width, label="SAE", color="#3C5488")
    bars2 = ax.bar(x + width / 2, trans_accs, width, label="Dialect", color="#E64B35")

    # Annotate
    for i in range(len(ranges)):
        ax.text(x[i] - width / 2, orig_accs[i] + 0.8, f"{orig_accs[i]:.1f}%",
                ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + width / 2, trans_accs[i] + 0.8, f"{trans_accs[i]:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # Gap annotations below the chart area
    for i in range(len(ranges)):
        ax.text(x[i], 2.5,
                f"gap: {gaps[i]:+.1f}pp  (n={counts[i]})",
                ha="center", va="bottom", fontsize=8.5, color="#666666")

    labels = [f"{r} feature{'s' if r != '1' else ''}" for r in ranges]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Feature Count Range (All Models Pooled)")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(orig_accs + trans_accs) + 8)

    fig.tight_layout()
    _save(fig, output_path)


def chart_feature_range_per_model(records: list[dict], output_path: Path):
    """Grouped bar chart: accuracy gap by feature range, one group per model."""
    _setup_style()

    benchmarks = ACCURACY_BENCHMARKS
    per_model = compute_per_model_range_stats(records, benchmarks)

    ranges = ["1", "2", "3", "4+"]
    models = [m for m in MODEL_ORDER if m in per_model]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_models = len(models)
    n_ranges = len(ranges)
    total_bars = n_models * n_ranges
    bar_width = 0.15
    group_width = n_ranges * bar_width + 0.1

    colors = ["#F5DEB3", "#E8A96A", "#D4792C", "#8B4513"]

    x = np.arange(n_models)
    for j, rng in enumerate(ranges):
        gaps = []
        for model in models:
            s = per_model[model].get(rng, {"gap": 0, "n": 0})
            gaps.append(s["gap"] if s["n"] >= 10 else np.nan)

        offset = (j - (n_ranges - 1) / 2) * bar_width
        bars = ax.bar(x + offset, gaps, bar_width, label=f"{rng} feature{'s' if rng != '1' else ''}",
                       color=colors[j])

        for i, (g, b) in enumerate(zip(gaps, bars)):
            if np.isnan(g):
                continue
            va = "top" if g < 0 else "bottom"
            off = -0.3 if g < 0 else 0.3
            ax.text(b.get_x() + b.get_width() / 2, g + off, f"{g:+.1f}",
                    ha="center", va=va, fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Accuracy Gap (pp)")
    ax.set_title("Accuracy Gap by Feature Count, Per Model")
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    _save(fig, output_path)


def chart_feature_range_per_dialect(records: list[dict], output_path: Path):
    """Accuracy gap by feature range, grouped by dialect."""
    _setup_style()

    dialect_order = ["aave", "hiberno_english", "indian_english"]
    ranges = ["1", "2", "3"]  # Skip 4+ due to tiny n

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bar_width = 0.22
    x = np.arange(len(dialect_order))

    colors = ["#F5DEB3", "#E8A96A", "#D4792C"]

    for j, rng in enumerate(ranges):
        gaps = []
        for dialect in dialect_order:
            subset = [r for r in records
                      if r["dialect"] == dialect
                      and r["benchmark"] in ACCURACY_BENCHMARKS
                      and r["feature_count"] == int(rng)]
            if len(subset) < 10:
                gaps.append(np.nan)
                continue
            orig = sum(1 for r in subset if r["original_correct"]) / len(subset)
            trans = sum(1 for r in subset if r["transformed_correct"]) / len(subset)
            gaps.append((trans - orig) * 100)

        offset = (j - 1) * bar_width
        bars = ax.bar(x + offset, gaps, bar_width,
                       label=f"{rng} feature{'s' if rng != '1' else ''}",
                       color=colors[j])

        for i, (g, b) in enumerate(zip(gaps, bars)):
            if np.isnan(g):
                continue
            va = "top" if g < 0 else "bottom"
            off = -0.3 if g < 0 else 0.3
            ax.text(b.get_x() + b.get_width() / 2, g + off, f"{g:+.1f}",
                    ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([format_dialect(d) for d in dialect_order], fontsize=11)
    ax.set_ylabel("Accuracy Gap (pp)")
    ax.set_title("Accuracy Gap by Feature Count, Per Dialect")
    ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, output_path)


def generate_feature_charts(
    audit_base: str = "data/audits",
    bench_dir: str = "data/benchmarks",
    output_dir: str = "data/reports/presentation",
):
    """Generate all feature analysis charts."""
    _setup_style()

    audit_path = Path(audit_base)
    bench_path = Path(bench_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading feature-accuracy data...")
    records = load_feature_accuracy_data(audit_path, bench_path)
    print(f"  Loaded {len(records)} pair records")

    # Quick stats
    from collections import Counter
    fc_dist = Counter(r["feature_count"] for r in records)
    print(f"  Feature count distribution: {dict(sorted(fc_dist.items()))}")

    print("\nThreshold stats (all models pooled):")
    stats = compute_threshold_stats(records)
    for t, s in sorted(stats.items()):
        print(f"  >= {t} features: n={s['n']:5d}  SAE={s['orig_acc']:.3f}  Dialect={s['trans_acc']:.3f}  gap={s['gap']:+.1f}pp")

    print("\nRange stats:")
    range_stats = compute_range_stats(records)
    for r, s in range_stats.items():
        print(f"  {r} features: n={s['n']:5d}  SAE={s['orig_acc']:.3f}  Dialect={s['trans_acc']:.3f}  gap={s['gap']:+.1f}pp")

    print("\nGenerating charts...")
    chart_feature_threshold(records, out / "feature_threshold.png")
    chart_feature_range_bars(records, out / "feature_range_bars.png")
    chart_feature_range_per_model(records, out / "feature_range_per_model.png")
    chart_feature_range_per_dialect(records, out / "feature_range_per_dialect.png")

    print("Done.")


if __name__ == "__main__":
    generate_feature_charts()
