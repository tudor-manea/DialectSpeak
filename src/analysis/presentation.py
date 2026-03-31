"""
Presentation-quality charts for thesis defense.

Produces 7 publication-ready PNG charts from audit data.
"""

import json
import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from .visualize import AuditData, load_audit

# --- Constants ---

DIALECT_COLORS = {
    "aave": "#E64B35",
    "hiberno_english": "#4DBBD5",
    "indian_english": "#00A087",
}
SAE_COLOR = "#3C5488"

ACCURACY_BENCHMARKS = [
    "gsm8k", "arc_challenge", "mmlu", "hellaswag", "boolq", "truthfulqa",
]
TOXICITY_BENCHMARKS = [
    "donotanswer", "toxigen", "realtoxicityprompts",
]

BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "boolq": "BoolQ",
    "truthfulqa": "TruthfulQA",
    "donotanswer": "DoNotAnswer",
    "toxigen": "ToxiGen",
    "realtoxicityprompts": "RealToxicity",
}

DIALECT_LABELS = {
    "aave": "AAVE",
    "hiberno_english": "Hiberno-English",
    "indian_english": "Indian English",
}

# Filename pattern: audit_{benchmark}_{dialect}_{model}_{YYYYMMDD}_{HHMMSS}.json
AUDIT_FILENAME_RE = re.compile(
    r"^audit_(.+?)_(aave|hiberno_english|indian_english)_(.+?)_(\d{8})_(\d{6})\.json$"
)
# Combined audit files: audit_{benchmark}_{dialect}_{model}_combined.json
AUDIT_COMBINED_RE = re.compile(
    r"^audit_(.+?)_(aave|hiberno_english|indian_english)_(.+?)_combined\.json$"
)


def format_benchmark(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name)


def format_dialect(name: str) -> str:
    return DIALECT_LABELS.get(name, name)


def _setup_style():
    """Configure matplotlib rcParams for presentation style."""
    # Try DM Sans, fall back to sans-serif
    font_path = None
    for f in fm.findSystemFonts():
        if "DMSans" in f or "DM-Sans" in f or "DM Sans" in f:
            font_path = f
            break
    if font_path:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
    else:
        font_name = "sans-serif"

    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def load_latest_audits(audit_dir: Path) -> list[AuditData]:
    """Load audits from directory, keeping only the latest per (benchmark, dialect).

    Combined files (from combine_results.py) always take priority over
    timestamped files.
    """
    groups: dict[tuple[str, str], tuple[str, Path]] = {}
    for path in audit_dir.glob("audit_*.json"):
        # Combined files get priority (sort after any timestamp)
        mc = AUDIT_COMBINED_RE.match(path.name)
        if mc:
            benchmark, dialect, _model = mc.groups()
            key = (benchmark, dialect)
            groups[key] = ("99999999999999", path)
            continue

        m = AUDIT_FILENAME_RE.match(path.name)
        if not m:
            continue
        benchmark, dialect, _model, date, time = m.groups()
        key = (benchmark, dialect)
        timestamp = date + time
        if key not in groups or timestamp > groups[key][0]:
            groups[key] = (timestamp, path)

    audits = []
    for (_benchmark, _dialect), (_ts, path) in sorted(groups.items()):
        try:
            audits.append(load_audit(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {path.name}: {e}")
    return audits


def load_llama_gsm8k(archive_dir: Path) -> list[AuditData]:
    """Load the 3 GSM8K llama3.1:8b audits from the archive."""
    audits = []
    for path in sorted(archive_dir.glob("audit_gsm8k_*.json")):
        try:
            audits.append(load_audit(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {path.name}: {e}")
    return audits


def _save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --- Chart 1: Accuracy Gap Heatmap ---

def chart_accuracy_gap_heatmap(audits: list[AuditData], output_path: Path):
    """6x3 diverging heatmap of accuracy gaps (accuracy benchmarks only)."""
    data = {
        (format_dialect(a.dialect), format_benchmark(a.benchmark)): a.accuracy_gap * 100
        for a in audits if a.benchmark in ACCURACY_BENCHMARKS
    }
    dialects = [format_dialect(d) for d in ["aave", "hiberno_english", "indian_english"]]
    benchmarks = [format_benchmark(b) for b in ACCURACY_BENCHMARKS]
    matrix = np.array([
        [data.get((d, b), np.nan) for b in benchmarks] for d in dialects
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = np.nanmax(np.abs(matrix))
    sns.heatmap(
        matrix, annot=True, fmt="+.1f", cmap="RdBu", center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=benchmarks, yticklabels=dialects,
        ax=ax, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Accuracy Gap (pp)", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Accuracy Gap by Dialect and Benchmark")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False)
    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 2: Dumbbell Chart ---

def chart_dumbbell(audits: list[AuditData], output_path: Path):
    """Dumbbell chart: SAE vs dialect accuracy for each benchmark-dialect combo."""
    acc_audits = [a for a in audits if a.benchmark in ACCURACY_BENCHMARKS]
    acc_audits.sort(key=lambda a: abs(a.accuracy_gap))

    labels = [
        f"{format_benchmark(a.benchmark)} / {format_dialect(a.dialect)}"
        for a in acc_audits
    ]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, a in enumerate(acc_audits):
        color = DIALECT_COLORS[a.dialect]
        ax.plot(
            [a.original_accuracy * 100, a.transformed_accuracy * 100], [i, i],
            color="#CCCCCC", linewidth=1.5, zorder=1,
        )
        ax.scatter(
            a.original_accuracy * 100, i, color=SAE_COLOR,
            s=60, zorder=2, edgecolors="white", linewidths=0.5,
        )
        ax.scatter(
            a.transformed_accuracy * 100, i, color=color,
            s=60, zorder=2, edgecolors="white", linewidths=0.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("SAE vs Dialect Accuracy Across Benchmarks")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Legend
    ax.scatter([], [], color=SAE_COLOR, s=60, label="SAE")
    for dialect, color in DIALECT_COLORS.items():
        ax.scatter([], [], color=color, s=60, label=format_dialect(dialect))
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 3: Dialect Summary Bars ---

def chart_dialect_summary(audits: list[AuditData], output_path: Path):
    """Grouped bars: mean SAE vs dialect accuracy per dialect (accuracy benchmarks)."""
    acc_audits = [a for a in audits if a.benchmark in ACCURACY_BENCHMARKS]
    dialect_order = ["aave", "hiberno_english", "indian_english"]

    means_sae, means_dialect = [], []
    for d in dialect_order:
        subset = [a for a in acc_audits if a.dialect == d]
        means_sae.append(np.mean([a.original_accuracy for a in subset]) * 100)
        means_dialect.append(np.mean([a.transformed_accuracy for a in subset]) * 100)

    x = np.arange(len(dialect_order))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.bar(x - width / 2, means_sae, width, label="SAE", color=SAE_COLOR)
    for i, d in enumerate(dialect_order):
        ax.bar(
            x[i] + width / 2, means_dialect[i], width,
            label=format_dialect(d), color=DIALECT_COLORS[d],
        )

    for i in range(len(dialect_order)):
        ax.text(
            x[i] - width / 2, means_sae[i] + 0.8,
            f"{means_sae[i]:.1f}%", ha="center", va="bottom", fontsize=10,
        )
        ax.text(
            x[i] + width / 2, means_dialect[i] + 0.8,
            f"{means_dialect[i]:.1f}%", ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([format_dialect(d) for d in dialect_order])
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Mean Accuracy by Dialect (Accuracy Benchmarks)")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, max(means_sae + means_dialect) + 8)

    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 4: Safety / Toxicity Subplots ---

def chart_toxicity_subplots(audits: list[AuditData], output_path: Path):
    """3 side-by-side grouped bar subplots for toxicity benchmarks."""
    tox_audits = [a for a in audits if a.benchmark in TOXICITY_BENCHMARKS]
    dialect_order = ["aave", "hiberno_english", "indian_english"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for idx, bench in enumerate(TOXICITY_BENCHMARKS):
        ax = axes[idx]
        subset = {a.dialect: a for a in tox_audits if a.benchmark == bench}
        x = np.arange(len(dialect_order))
        width = 0.32

        sae_vals = [subset[d].original_accuracy * 100 if d in subset else 0 for d in dialect_order]
        dia_vals = [subset[d].transformed_accuracy * 100 if d in subset else 0 for d in dialect_order]

        ax.bar(
            x - width / 2, sae_vals, width, label="SAE", color=SAE_COLOR,
        )
        ax.bar(
            x + width / 2, dia_vals, width, label="Dialect",
            color=[DIALECT_COLORS[d] for d in dialect_order],
        )

        ax.set_xticks(x)
        ax.set_xticklabels([format_dialect(d) for d in dialect_order], fontsize=10, rotation=25, ha="right")
        ax.set_title(format_benchmark(bench), fontsize=14)
        ax.set_ylabel("Rate (%)" if idx == 0 else "")
        if idx == 0:
            ax.legend(fontsize=10)

    fig.suptitle("Safety Benchmark Performance by Dialect", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 5: Toxicity Gap Heatmap ---

def chart_toxicity_gap_heatmap(audits: list[AuditData], output_path: Path):
    """Compact 3x3 diverging heatmap for toxicity benchmarks."""
    data = {
        (format_dialect(a.dialect), format_benchmark(a.benchmark)): a.accuracy_gap * 100
        for a in audits if a.benchmark in TOXICITY_BENCHMARKS
    }
    dialects = [format_dialect(d) for d in ["aave", "hiberno_english", "indian_english"]]
    benchmarks = [format_benchmark(b) for b in TOXICITY_BENCHMARKS]
    matrix = np.array([
        [data.get((d, b), np.nan) for b in benchmarks] for d in dialects
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.nanmax(np.abs(matrix))
    sns.heatmap(
        matrix, annot=True, fmt="+.1f", cmap="RdBu", center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=benchmarks, yticklabels=dialects,
        ax=ax, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Rate Gap (pp)", "shrink": 0.8},
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    ax.set_title("Toxicity Benchmark Gap by Dialect")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False)
    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 6: Cross-Model GSM8K Comparison ---

def chart_cross_model_gsm8k(
    qwen_audits: list[AuditData],
    llama_audits: list[AuditData],
    output_path: Path,
):
    """Grouped bar: llama3.1:8b vs qwen2.5:14b dialect gaps on GSM8K."""
    dialect_order = ["aave", "hiberno_english", "indian_english"]

    qwen_gaps = {a.dialect: a.accuracy_gap * 100 for a in qwen_audits if a.benchmark == "gsm8k"}
    llama_gaps = {a.dialect: a.accuracy_gap * 100 for a in llama_audits if a.benchmark == "gsm8k"}

    x = np.arange(len(dialect_order))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5.5))

    llama_vals = [llama_gaps.get(d, 0) for d in dialect_order]
    qwen_vals = [qwen_gaps.get(d, 0) for d in dialect_order]

    bars_l = ax.bar(x - width / 2, llama_vals, width, label="Llama 3.1 8B", color="#7E6148")
    bars_q = ax.bar(x + width / 2, qwen_vals, width, label="Qwen 2.5 14B", color="#B09C85")

    for bar in list(bars_l) + list(bars_q):
        h = bar.get_height()
        va = "top" if h < 0 else "bottom"
        offset = -0.5 if h < 0 else 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + offset,
            f"{h:+.1f}", ha="center", va=va, fontsize=11,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([format_dialect(d) for d in dialect_order])
    ax.set_ylabel("Accuracy Gap (pp)")
    ax.set_title("GSM8K Dialect Gap: Llama 3.1 8B vs Qwen 2.5 14B")
    ax.legend(fontsize=11)

    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 7: Overall Bias Summary ---

def chart_overall_bias(audits: list[AuditData], output_path: Path):
    """Horizontal bar chart: average gap per dialect, split accuracy vs toxicity."""
    dialect_order = ["aave", "hiberno_english", "indian_english"]

    acc_gaps, tox_gaps = {}, {}
    for d in dialect_order:
        acc_subset = [a for a in audits if a.dialect == d and a.benchmark in ACCURACY_BENCHMARKS]
        tox_subset = [a for a in audits if a.dialect == d and a.benchmark in TOXICITY_BENCHMARKS]
        acc_gaps[d] = np.mean([a.accuracy_gap for a in acc_subset]) * 100 if acc_subset else 0
        tox_gaps[d] = np.mean([a.accuracy_gap for a in tox_subset]) * 100 if tox_subset else 0

    y = np.arange(len(dialect_order))
    height = 0.32
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.barh(
        y - height / 2, [acc_gaps[d] for d in dialect_order], height,
        label="Accuracy Benchmarks", color="#4DBBD5",
    )
    ax.barh(
        y + height / 2, [tox_gaps[d] for d in dialect_order], height,
        label="Toxicity Benchmarks", color="#E64B35",
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([format_dialect(d) for d in dialect_order])
    ax.set_xlabel("Mean Gap (pp)")
    ax.set_title("Overall Dialect Bias: Accuracy vs Toxicity")
    ax.legend(fontsize=11, loc="lower left")

    # Annotate values
    for i, d in enumerate(dialect_order):
        for val, offset in [(acc_gaps[d], -height / 2), (tox_gaps[d], height / 2)]:
            ha = "right" if val < 0 else "left"
            nudge = -0.3 if val < 0 else 0.3
            ax.text(val + nudge, i + offset, f"{val:+.1f}", ha=ha, va="center", fontsize=10)

    fig.tight_layout()
    _save(fig, output_path)


# --- Chart 8: Multi-Model Accuracy Table ---

MODEL_LABELS = {
    "qwen2.5:14b": "Qwen 2.5 14B",
    "qwen2.5:7b": "Qwen 2.5 7B",
    "llama3.1:8b": "Llama 3.1 8B",
    "gemma2:9b": "Gemma 2 9B",
    "mistral:7b": "Mistral 7B",
}

MODEL_ORDER = ["qwen2.5:14b", "qwen2.5:7b", "gemma2:9b", "llama3.1:8b", "mistral:7b"]


def _load_all_model_audits(base_dir: Path) -> list[AuditData]:
    """Load latest audits from all model subdirectories."""
    model_dirs = {
        "qwen2.5_14b", "qwen2.5_7b", "llama3.1_8b", "gemma2_9b", "mistral_7b",
    }
    all_audits = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name in model_dirs:
            audits = load_latest_audits(d)
            all_audits.extend(audits)
    return all_audits


def chart_multi_model_table(
    audits: list[AuditData],
    output_path: Path,
    benchmarks: list[str] | None = None,
    title: str = "Model Accuracy by Dialect (%)",
):
    """Color-coded table: rows = benchmark x model, cols = SAE + dialects."""
    if benchmarks is None:
        benchmarks = ACCURACY_BENCHMARKS

    dialect_order = ["aave", "hiberno_english", "indian_english"]
    col_headers = ["SAE", "Mean", "AAVE", "Hiberno-Eng", "Indian Eng"]

    # Build lookup: (benchmark, model, dialect) -> (orig_acc, trans_acc)
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for a in audits:
        if a.benchmark in benchmarks:
            lookup[(a.benchmark, a.model, a.dialect)] = (
                a.original_accuracy,
                a.transformed_accuracy,
            )

    # Discover models present in data, ordered
    models_present = [m for m in MODEL_ORDER if any(
        k[1] == m for k in lookup
    )]

    # Build row data: list of (benchmark_label, model_label, [sae, mean, aave, hib, ind])
    rows: list[tuple[str, str, list[float | None]]] = []
    for bench in benchmarks:
        for model in models_present:
            sae_vals = [
                lookup[(bench, model, d)][0]
                for d in dialect_order
                if (bench, model, d) in lookup
            ]
            sae = np.mean(sae_vals) * 100 if sae_vals else None

            dialect_accs = []
            for d in dialect_order:
                key = (bench, model, d)
                if key in lookup:
                    dialect_accs.append(lookup[key][1] * 100)
                else:
                    dialect_accs.append(None)

            valid = [v for v in dialect_accs if v is not None]
            mean_dialect = np.mean(valid) if valid else None

            row_vals = [sae, mean_dialect] + dialect_accs
            rows.append((format_benchmark(bench), MODEL_LABELS.get(model, model), row_vals))

    n_rows = len(rows)
    n_models = len(models_present)

    # Collect all numeric values for color scaling
    all_vals = []
    for _, _, vals in rows:
        for v in vals:
            if v is not None:
                all_vals.append(v)
    vmin, vmax = min(all_vals), max(all_vals)

    # Color map: orange (low) -> white (mid) -> blue (high)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("table_cmap", [
        "#D4792C",   # dark orange
        "#E8A96A",   # light orange
        "#F5DEB3",   # wheat
        "#FFFFFF",   # white
        "#B3D4F0",   # light blue
        "#6AAED6",   # medium blue
        "#3A7EC0",   # blue
    ])

    # Layout constants
    row_h = 0.32
    bench_col_w = 1.1
    model_col_w = 1.3
    data_col_w = 1.0
    n_data_cols = len(col_headers)
    total_w = bench_col_w + model_col_w + n_data_cols * data_col_w
    header_h = 0.45
    group_sep = 0.12
    n_bench_groups = len(benchmarks)
    total_h = header_h + n_rows * row_h + n_bench_groups * group_sep + 0.4

    fig, ax = plt.subplots(figsize=(total_w * 1.1, total_h * 1.0))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Column x-positions
    data_x0 = bench_col_w + model_col_w
    col_xs = [data_x0 + i * data_col_w for i in range(n_data_cols)]

    # Header row
    y = total_h - 0.2
    ax.text(bench_col_w / 2, y, "", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(bench_col_w + model_col_w / 2, y, "Model", ha="center", va="center", fontsize=9.5, fontweight="bold")
    for i, label in enumerate(col_headers):
        ax.text(
            col_xs[i] + data_col_w / 2, y, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#333333",
        )

    y -= 0.22
    ax.plot([0, total_w], [y, y], color="#555555", linewidth=1.0)

    # Draw rows
    current_bench = None
    row_counter = 0
    for bench_label, model_label, vals in rows:
        # Benchmark group separator
        if bench_label != current_bench:
            if current_bench is not None:
                y -= group_sep
                ax.plot([0, total_w], [y + group_sep / 2, y + group_sep / 2], color="#DDDDDD", linewidth=0.5)
            current_bench = bench_label
            row_counter = 0

        y -= row_h

        # Benchmark label (only on first row of group)
        if row_counter == 0:
            bench_y = y + (n_models - 1) * row_h / 2
            ax.text(
                bench_col_w / 2, bench_y, bench_label,
                ha="center", va="center", fontsize=10, fontweight="bold", color="#1a1a1a",
            )

        # Alternating row background
        if row_counter % 2 == 1:
            rect = plt.Rectangle(
                (bench_col_w, y - row_h / 2 + 0.02), total_w - bench_col_w, row_h - 0.02,
                facecolor="#F8F8F8", edgecolor="none",
            )
            ax.add_patch(rect)

        # Model name
        ax.text(
            bench_col_w + model_col_w / 2, y, model_label,
            ha="center", va="center", fontsize=8.5, color="#444444",
        )

        # Data cells
        for col_idx, val in enumerate(vals):
            cx = col_xs[col_idx] + data_col_w / 2

            if val is None:
                ax.text(cx, y, "-", ha="center", va="center", fontsize=8.5, color="#BBBBBB")
                continue

            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = cmap(norm_val)

            rect = plt.Rectangle(
                (col_xs[col_idx] + 0.04, y - row_h / 2 + 0.03),
                data_col_w - 0.08, row_h - 0.05,
                facecolor=color, edgecolor="none", alpha=0.9,
            )
            ax.add_patch(rect)

            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "#1a1a1a" if luminance > 0.45 else "#FFFFFF"

            ax.text(
                cx, y, f"{val:.1f}",
                ha="center", va="center", fontsize=8.5, fontweight="medium", color=text_color,
            )

        row_counter += 1

    # Bottom line
    ax.plot([0, total_w], [y - row_h / 2, y - row_h / 2], color="#DDDDDD", linewidth=0.5)

    # Title
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(pad=0.3)
    _save(fig, output_path)


def chart_multi_model_table_toxicity(
    audits: list[AuditData], output_path: Path,
):
    """Same table format but for toxicity benchmarks (refusal rate)."""
    chart_multi_model_table(
        audits, output_path,
        benchmarks=TOXICITY_BENCHMARKS,
        title="Model Refusal Rate by Dialect (%)",
    )


# --- Chart 9: McNemar Significance Heatmap ---

def chart_significance_heatmap(audits: list[AuditData], output_path: Path):
    """Heatmap of McNemar's test p-values across all models, benchmarks, dialects.

    Uses BH-corrected p-values for significance markers.
    """
    from .significance import test_all_audits, apply_bh_correction

    results = test_all_audits(audits)
    apply_bh_correction(results)

    # Build lookup: (model, benchmark, dialect) -> McNemarResult
    sig_lookup = {(r.model, r.benchmark, r.dialect): r for r in results}

    dialect_order = ["aave", "hiberno_english", "indian_english"]
    all_benchmarks = ACCURACY_BENCHMARKS + TOXICITY_BENCHMARKS
    models_present = [m for m in MODEL_ORDER if any(r.model == m for r in results)]

    # Build row labels and p-value matrix
    row_labels = []
    matrix = []
    gap_matrix = []
    for bench in all_benchmarks:
        for model in models_present:
            label = f"{format_benchmark(bench)} / {MODEL_LABELS.get(model, model)}"
            row_labels.append(label)
            row = []
            gap_row = []
            for d in dialect_order:
                key = (model, bench, d)
                if key in sig_lookup:
                    row.append(sig_lookup[key].p_value)
                    # Get accuracy gap for annotation
                    audit = next(
                        (a for a in audits
                         if a.model == model and a.benchmark == bench and a.dialect == d),
                        None,
                    )
                    gap_row.append(audit.accuracy_gap * 100 if audit else 0)
                else:
                    row.append(np.nan)
                    gap_row.append(0)
            matrix.append(row)
            gap_matrix.append(gap_row)

    matrix = np.array(matrix)
    gap_matrix = np.array(gap_matrix)

    # That's too many rows. Instead, make a compact summary: benchmark x dialect
    # with one heatmap per model, or aggregate across models.
    # Better approach: compact table showing gap + significance marker per (model, benchmark, dialect)

    # Build a simpler grid: rows = benchmark, columns = model x dialect
    # Actually, let's do: rows = benchmark x model (like the accuracy table),
    # cols = 3 dialects, cell = gap with significance star

    row_labels = []
    p_vals = []
    gaps = []
    for bench in all_benchmarks:
        for model in models_present:
            row_labels.append(f"{MODEL_LABELS.get(model, model)}")
            p_row = []
            gap_row = []
            for d in dialect_order:
                key = (model, bench, d)
                if key in sig_lookup:
                    p_row.append(sig_lookup[key].p_value)
                else:
                    p_row.append(np.nan)
                audit = next(
                    (a for a in audits
                     if a.model == model and a.benchmark == bench and a.dialect == d),
                    None,
                )
                gap_row.append(audit.accuracy_gap * 100 if audit else np.nan)
            p_vals.append(p_row)
            gaps.append(gap_row)

    p_matrix = np.array(p_vals)
    gap_matrix = np.array(gaps)
    n_rows = len(row_labels)

    # Color: diverging by gap, stars for significance
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("gap_sig", [
        "#C0392B",   # strong red (negative gap)
        "#E8A09A",   # light red
        "#FFFFFF",   # zero
        "#A3D5A3",   # light green
        "#27AE60",   # strong green (positive gap)
    ])

    col_labels = [format_dialect(d) for d in dialect_order]

    # Figure
    row_h = 0.30
    bench_sep = 0.25
    fig_h = 1.0 + n_rows * row_h + len(all_benchmarks) * bench_sep
    fig_w = 8.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Column layout
    bench_w = 1.0
    model_w = 1.3
    data_x0 = bench_w + model_w
    data_w = (fig_w - data_x0) / 3

    # Header
    y = fig_h - 0.3
    ax.text(bench_w / 2, y, "", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(bench_w + model_w / 2, y, "Model", ha="center", va="center", fontsize=9.5, fontweight="bold")
    for i, label in enumerate(col_labels):
        ax.text(
            data_x0 + (i + 0.5) * data_w, y, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
        )
    y -= 0.2
    ax.plot([0, fig_w], [y, y], color="#555555", linewidth=1.0)

    # Normalize gaps for coloring
    valid_gaps = gap_matrix[~np.isnan(gap_matrix)]
    if len(valid_gaps) > 0:
        abs_max = max(abs(valid_gaps.min()), abs(valid_gaps.max()))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)

    current_bench_idx = -1
    row_in_group = 0
    for row_idx in range(n_rows):
        bench_idx = row_idx // len(models_present)

        if bench_idx != current_bench_idx:
            current_bench_idx = bench_idx
            row_in_group = 0
            if bench_idx > 0:
                y -= bench_sep
                ax.plot([0, fig_w], [y + bench_sep / 2, y + bench_sep / 2], color="#DDDDDD", linewidth=0.5)
            # Benchmark label
            bench_name = format_benchmark(all_benchmarks[bench_idx])
            bench_y = y - (len(models_present) * row_h) / 2
            ax.text(
                bench_w / 2, bench_y, bench_name,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
            )

        y -= row_h

        # Model name
        ax.text(
            bench_w + model_w / 2, y, row_labels[row_idx],
            ha="center", va="center", fontsize=8, color="#444444",
        )

        # Data cells
        for col_idx in range(3):
            cx = data_x0 + (col_idx + 0.5) * data_w
            gap = gap_matrix[row_idx, col_idx]
            p = p_matrix[row_idx, col_idx]

            if np.isnan(gap):
                ax.text(cx, y, "-", ha="center", va="center", fontsize=8.5, color="#BBBBBB")
                continue

            # Cell background colored by gap
            color = cmap(norm(gap))
            rect = plt.Rectangle(
                (data_x0 + col_idx * data_w + 0.04, y - row_h / 2 + 0.02),
                data_w - 0.08, row_h - 0.04,
                facecolor=color, edgecolor="none", alpha=0.85,
            )
            ax.add_patch(rect)

            # Text: gap value + significance stars (BH-corrected)
            key_sig = (model, bench, d) if 'model' in dir() else None
            r_match = sig_lookup.get((models_present[row_idx % len(models_present)],
                                      all_benchmarks[row_idx // len(models_present)],
                                      dialect_order[col_idx]))
            sig_marker = ""
            if r_match:
                if r_match.bh_significant_01:
                    sig_marker = "**"
                elif r_match.bh_significant_05:
                    sig_marker = "*"

            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "#1a1a1a" if luminance > 0.45 else "#FFFFFF"

            ax.text(
                cx, y, f"{gap:+.1f}{sig_marker}",
                ha="center", va="center", fontsize=8, fontweight="medium", color=text_color,
            )

        row_in_group += 1

    ax.plot([0, fig_w], [y - row_h / 2, y - row_h / 2], color="#DDDDDD", linewidth=0.5)

    # Legend for significance markers
    ax.text(
        fig_w / 2, 0.12,
        "* FDR < 0.05    ** FDR < 0.01    (McNemar's test, BH-corrected)",
        ha="center", va="center", fontsize=8.5, color="#666666", style="italic",
    )

    fig.suptitle("Accuracy Gap (pp) with Statistical Significance", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(pad=0.3)
    _save(fig, output_path)


# --- Orchestration ---

def generate_all(
    audit_base: str = "data/audits",
    output_dir: str = "data/reports/presentation",
):
    """Generate all presentation charts."""
    base_path = Path(audit_base)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _setup_style()

    # Load all audits across all models
    print("Loading all model audits...")
    all_audits = _load_all_model_audits(base_path)
    print(f"  Loaded {len(all_audits)} audits across all models")

    # Single-model audits for existing charts (qwen as primary)
    qwen_audits = load_latest_audits(base_path / "qwen2.5_14b")
    llama_audits = load_latest_audits(base_path / "llama3.1_8b")
    print(f"  Qwen: {len(qwen_audits)}, Llama: {len(llama_audits)}")

    print("\nGenerating charts...")
    chart_multi_model_table(all_audits, out / "multi_model_accuracy.png")
    chart_multi_model_table_toxicity(all_audits, out / "multi_model_toxicity.png")
    chart_significance_heatmap(all_audits, out / "significance_heatmap.png")
    chart_accuracy_gap_heatmap(qwen_audits, out / "accuracy_gap_heatmap.png")
    chart_dumbbell(qwen_audits, out / "dumbbell.png")
    chart_dialect_summary(qwen_audits, out / "dialect_summary.png")
    chart_toxicity_subplots(qwen_audits, out / "toxicity_subplots.png")
    chart_toxicity_gap_heatmap(qwen_audits, out / "toxicity_gap_heatmap.png")
    chart_cross_model_gsm8k(qwen_audits, llama_audits, out / "cross_model_gsm8k.png")
    chart_overall_bias(qwen_audits, out / "overall_bias.png")

    # Feature analysis charts
    from .feature_analysis import generate_feature_charts
    generate_feature_charts(
        audit_base=str(base_path),
        bench_dir="data/benchmarks",
        output_dir=str(out),
    )

    print(f"\nDone. {len(list(out.glob('*.png')))} charts in {out}/")


if __name__ == "__main__":
    generate_all()
