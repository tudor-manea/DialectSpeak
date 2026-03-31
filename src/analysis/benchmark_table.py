"""Generate HELM-style benchmark tables showing model performance across dialects."""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch


def load_audit_data(audit_dir="data/audits"):
    """Load all audit results into a list of dicts.

    Prefers combined files over timestamped ones for the same
    benchmark-dialect pair.
    """
    model_dirs = ["qwen2.5_14b", "qwen2.5_7b", "gemma2_9b", "llama3.1_8b", "mistral_7b"]
    all_data = []
    for model_dir in model_dirs:
        path = os.path.join(audit_dir, model_dir)
        # Deduplicate: combined files take priority
        best = {}
        for f in sorted(glob.glob(os.path.join(path, "audit_*.json"))):
            fname = os.path.basename(f)
            d = json.load(open(f))
            key = (d["benchmark"], d["dialect"])
            if "_combined.json" in fname or key not in best:
                best[key] = d
        for d in best.values():
            all_data.append({
                "benchmark": d["benchmark"],
                "dialect": d["dialect"],
                "model": d["model"],
                "original_accuracy": d["original_accuracy"],
                "transformed_accuracy": d["transformed_accuracy"],
                "total_pairs": d["total_pairs"],
            })
    return all_data


BENCHMARK_NAMES = {
    "mmlu": "MMLU",
    "arc_challenge": "ARC",
    "truthfulqa": "TruthfulQA",
    "gsm8k": "GSM8K",
    "hellaswag": "HellaSwag",
    "boolq": "BoolQ",
    "realtoxicityprompts": "RTP",
    "donotanswer": "DoNotAnswer",
    "toxigen": "ToxiGen",
}

QA_BENCHMARKS = ["mmlu", "arc_challenge", "truthfulqa", "gsm8k", "hellaswag", "boolq"]
SAFETY_BENCHMARKS = ["realtoxicityprompts", "donotanswer", "toxigen"]

MODEL_NAMES = {
    "qwen2.5:14b": "Qwen 2.5 14B",
    "qwen2.5:7b": "Qwen 2.5 7B",
    "gemma2:9b": "Gemma 2 9B",
    "llama3.1:8b": "Llama 3.1 8B",
    "mistral:7b": "Mistral 7B",
}

MODEL_ORDER = ["qwen2.5:14b", "qwen2.5:7b", "gemma2:9b", "llama3.1:8b", "mistral:7b"]

DIALECT_ORDER = ["aave", "hiberno_english", "indian_english"]


def build_table_data(all_data, benchmarks):
    """Build the table: rows = (benchmark, model), cols = [SAE, Mean, AAVE, HIB, IND]."""
    lookup = {}
    for d in all_data:
        key = (d["benchmark"], d["model"], d["dialect"])
        lookup[key] = d

    rows = []
    for bm in benchmarks:
        for model in MODEL_ORDER:
            sae_vals = []
            dialect_vals = {}
            for dialect in DIALECT_ORDER:
                key = (bm, model, dialect)
                if key in lookup:
                    sae_vals.append(lookup[key]["original_accuracy"] * 100)
                    dialect_vals[dialect] = lookup[key]["transformed_accuracy"] * 100
            sae = np.mean(sae_vals) if sae_vals else 0
            mean_dialect = np.mean(list(dialect_vals.values())) if dialect_vals else 0
            row = {
                "benchmark": bm,
                "model": model,
                "SAE": round(sae, 1),
                "Mean": round(mean_dialect, 1),
            }
            for dialect in DIALECT_ORDER:
                row[dialect] = round(dialect_vals.get(dialect, 0), 1)
            rows.append(row)
    return rows


def render_table(rows, benchmarks, output_path, title=None, footnote=None):
    """Render the benchmark table as a matplotlib figure."""
    col_keys = ["SAE", "Mean", "aave", "hiberno_english", "indian_english"]
    col_labels = ["SAE", "Mean", "AAVE", "HIB", "IND"]
    n_cols = len(col_labels)
    n_rows = len(rows)
    n_benchmarks = len(benchmarks)

    # Color maps
    cmap_orange = mcolors.LinearSegmentedColormap.from_list(
        "warm", ["#fff8ef", "#fde8c9", "#f5c061", "#e8943a"]
    )
    cmap_blue = mcolors.LinearSegmentedColormap.from_list(
        "cool", ["#f0f7ff", "#c8ddf5", "#8bb8e8"]
    )

    # Figure sizing
    bm_col_width = 1.6
    model_col_width = 1.8
    data_col_width = 0.9
    row_height = 0.38
    header_height = 0.5
    title_height = 0.5 if title else 0
    footnote_height = 0.4 if footnote else 0

    fig_width = bm_col_width + model_col_width + n_cols * data_col_width + 0.4
    fig_height = (title_height + header_height + n_rows * row_height
                  + (n_benchmarks - 1) * 0.08 + footnote_height + 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Starting positions
    x_bm = 0.15
    x_model = x_bm + bm_col_width
    x_data_start = x_model + model_col_width
    y_top = fig_height - 0.3

    # Title
    if title:
        ax.text(fig_width / 2, y_top, title, ha="center", va="center",
                fontsize=12, fontweight="bold", fontfamily="sans-serif")
        y_top -= title_height

    # Draw header
    for i, label in enumerate(col_labels):
        x = x_data_start + i * data_col_width + data_col_width / 2
        ax.text(x, y_top, label, ha="center", va="center",
                fontsize=10, fontweight="bold", fontfamily="sans-serif")

    # Horizontal line under header
    y_line = y_top - header_height / 2
    ax.plot([x_bm - 0.1, fig_width - 0.15], [y_line, y_line],
            color="#aaaaaa", linewidth=0.8)

    # Vertical black line separating SAE from dialect columns
    x_vsep = x_data_start + data_col_width
    y_bottom = fig_height - title_height - header_height - n_rows * row_height - (n_benchmarks - 1) * 0.08 - 0.3
    ax.plot([x_vsep, x_vsep], [y_top + 0.2, y_bottom],
            color="#333333", linewidth=1.0)

    # Draw rows
    y = y_line - row_height / 2 - 0.08
    prev_bm = None
    for row in rows:
        bm = row["benchmark"]

        if bm != prev_bm:
            if prev_bm is not None:
                y -= 0.08
                y_sep = y + row_height / 2 + 0.04
                ax.plot([x_bm - 0.1, fig_width - 0.15], [y_sep, y_sep],
                        color="#cccccc", linewidth=0.5)

            group_size = sum(1 for r in rows if r["benchmark"] == bm)
            bm_center_y = y - (group_size - 1) * row_height / 2
            ax.text(x_bm, bm_center_y, BENCHMARK_NAMES[bm],
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    fontfamily="sans-serif")
            prev_bm = bm

        ax.text(x_model, y, MODEL_NAMES[row["model"]],
                ha="left", va="center", fontsize=9, fontfamily="sans-serif",
                color="#444444")

        sae_val = row["SAE"]
        for i, col in enumerate(col_keys):
            val = row[col]
            x_cell = x_data_start + i * data_col_width

            if col == "SAE":
                color = "#ffffff"
            elif sae_val == 0:
                color = "#f5f5f5"
            elif val > sae_val:
                gap = min((val - sae_val) / sae_val, 0.5)
                color = cmap_blue(gap / 0.5)
            else:
                gap = min((sae_val - val) / sae_val, 0.5)
                color = cmap_orange(gap / 0.5)

            rect = FancyBboxPatch(
                (x_cell + 0.04, y - row_height / 2 + 0.04),
                data_col_width - 0.08,
                row_height - 0.08,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
            )
            ax.add_patch(rect)

            ax.text(x_cell + data_col_width / 2, y, f"{val:.1f}",
                    ha="center", va="center", fontsize=8.5,
                    fontfamily="monospace",
                    color="#333333" if val > 5 else "#999999")

        y -= row_height

    if footnote:
        ax.text(x_bm, y - 0.15, footnote,
                ha="left", va="center", fontsize=7.5, fontstyle="italic",
                color="#666666", fontfamily="sans-serif")

    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none", pad_inches=0.15)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    data = load_audit_data()

    # QA benchmarks
    qa_rows = build_table_data(data, QA_BENCHMARKS)
    render_table(
        qa_rows, QA_BENCHMARKS,
        output_path="data/reports/benchmark_table_qa.png",
        title="QA Benchmark Accuracy (%)",
    )

    # Safety benchmarks
    safety_rows = build_table_data(data, SAFETY_BENCHMARKS)
    render_table(
        safety_rows, SAFETY_BENCHMARKS,
        output_path="data/reports/benchmark_table_safety.png",
        title="Safety Benchmark Refusal Rate (%)",
        footnote="Scores show refusal rate: higher = model refused more toxic/harmful prompts (safer).",
    )
