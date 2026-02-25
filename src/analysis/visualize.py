"""
Visualization Module

Creates heatmaps, charts, and reports from fairness audit results.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class AuditData:
    """Parsed audit result data."""

    benchmark: str
    dialect: str
    model: str
    total_pairs: int
    original_accuracy: float
    transformed_accuracy: float
    accuracy_gap: float
    both_correct: int
    both_wrong: int
    original_only_correct: int
    transformed_only_correct: int
    benchmark_type: str = "numerical"

    @classmethod
    def from_json(cls, data: dict) -> "AuditData":
        """Create AuditData from JSON dict."""
        return cls(
            benchmark=data["benchmark"],
            dialect=data["dialect"],
            model=data["model"],
            total_pairs=data["total_pairs"],
            original_accuracy=data["original_accuracy"],
            transformed_accuracy=data["transformed_accuracy"],
            accuracy_gap=data["accuracy_gap"],
            both_correct=data["both_correct"],
            both_wrong=data["both_wrong"],
            original_only_correct=data["original_only_correct"],
            transformed_only_correct=data["transformed_only_correct"],
            benchmark_type=data.get("benchmark_type", "numerical"),
        )


def load_audit(path: Path) -> AuditData:
    """Load a single audit result from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuditData.from_json(data)


def load_all_audits(
    audit_dir: Path, exclude_samples: bool = True
) -> list[AuditData]:
    """
    Load all audit results from a directory.

    Args:
        audit_dir: Directory containing audit JSON files
        exclude_samples: Whether to exclude sample/test files

    Returns:
        List of AuditData objects
    """
    audits = []
    for path in sorted(audit_dir.glob("audit_*.json")):
        if exclude_samples and "sample" in path.name:
            continue
        try:
            audits.append(load_audit(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {path.name}: {e}")
    return audits


def _audits_to_dataframe(audits: list[AuditData]) -> pd.DataFrame:
    """Convert list of AuditData to DataFrame."""
    records = []
    for a in audits:
        records.append(
            {
                "benchmark": a.benchmark,
                "dialect": a.dialect,
                "model": a.model,
                "total_pairs": a.total_pairs,
                "original_accuracy": a.original_accuracy,
                "transformed_accuracy": a.transformed_accuracy,
                "accuracy_gap": a.accuracy_gap,
                "both_correct": a.both_correct,
                "both_wrong": a.both_wrong,
                "original_only_correct": a.original_only_correct,
                "transformed_only_correct": a.transformed_only_correct,
            }
        )
    return pd.DataFrame(records)


def create_accuracy_heatmap(
    audits: list[AuditData],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 8),
    show: bool = False,
) -> plt.Figure:
    """
    Create heatmap of original vs transformed accuracy by dialect × benchmark.

    Args:
        audits: List of audit results
        output_path: Path to save the figure
        figsize: Figure size (width, height)
        show: Whether to display the figure

    Returns:
        matplotlib Figure
    """
    df = _audits_to_dataframe(audits)

    # Create pivot tables for original and transformed accuracy
    orig_pivot = df.pivot(
        index="dialect", columns="benchmark", values="original_accuracy"
    )
    trans_pivot = df.pivot(
        index="dialect", columns="benchmark", values="transformed_accuracy"
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original accuracy heatmap
    sns.heatmap(
        orig_pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[0],
        cbar_kws={"label": "Accuracy"},
    )
    axes[0].set_title("Original (SAE) Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Benchmark", fontsize=12)
    axes[0].set_ylabel("Dialect", fontsize=12)

    # Transformed accuracy heatmap
    sns.heatmap(
        trans_pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_kws={"label": "Accuracy"},
    )
    axes[1].set_title("Transformed (Dialect) Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Benchmark", fontsize=12)
    axes[1].set_ylabel("Dialect", fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def create_accuracy_gap_chart(
    audits: list[AuditData],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = False,
) -> plt.Figure:
    """
    Create heatmap showing accuracy gaps (transformed - original).

    Negative values = dialect underperforms SAE
    Positive values = dialect outperforms SAE

    Args:
        audits: List of audit results
        output_path: Path to save the figure
        figsize: Figure size
        show: Whether to display the figure

    Returns:
        matplotlib Figure
    """
    df = _audits_to_dataframe(audits)

    # Create pivot table for accuracy gap
    gap_pivot = df.pivot(index="dialect", columns="benchmark", values="accuracy_gap")

    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0
    max_gap = max(abs(gap_pivot.min().min()), abs(gap_pivot.max().max()))
    sns.heatmap(
        gap_pivot,
        annot=True,
        fmt="+.1%",
        cmap="RdBu",
        center=0,
        vmin=-max_gap,
        vmax=max_gap,
        ax=ax,
        cbar_kws={"label": "Accuracy Gap"},
    )
    ax.set_title(
        "Accuracy Gap (Dialect - SAE)\nNegative = Dialect Underperforms",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Dialect", fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def create_dialect_comparison(
    audits: list[AuditData],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 6),
    show: bool = False,
) -> plt.Figure:
    """
    Create grouped bar chart comparing original vs transformed accuracy per dialect.

    Args:
        audits: List of audit results
        output_path: Path to save the figure
        figsize: Figure size
        show: Whether to display the figure

    Returns:
        matplotlib Figure
    """
    df = _audits_to_dataframe(audits)

    # Aggregate by dialect (mean across benchmarks)
    dialect_stats = (
        df.groupby("dialect")
        .agg(
            {
                "original_accuracy": "mean",
                "transformed_accuracy": "mean",
                "accuracy_gap": "mean",
                "total_pairs": "sum",
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(dialect_stats))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        dialect_stats["original_accuracy"],
        width,
        label="Original (SAE)",
        color="#2ecc71",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        dialect_stats["transformed_accuracy"],
        width,
        label="Transformed (Dialect)",
        color="#3498db",
        edgecolor="black",
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Dialect", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Average Accuracy by Dialect (Across All Benchmarks)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dialect_stats["dialect"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def create_benchmark_comparison(
    audits: list[AuditData],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 6),
    show: bool = False,
) -> plt.Figure:
    """
    Create grouped bar chart comparing accuracy across benchmarks.

    Args:
        audits: List of audit results
        output_path: Path to save the figure
        figsize: Figure size
        show: Whether to display the figure

    Returns:
        matplotlib Figure
    """
    df = _audits_to_dataframe(audits)

    # Aggregate by benchmark (mean across dialects)
    benchmark_stats = (
        df.groupby("benchmark")
        .agg(
            {
                "original_accuracy": "mean",
                "transformed_accuracy": "mean",
                "accuracy_gap": "mean",
                "total_pairs": "sum",
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(benchmark_stats))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        benchmark_stats["original_accuracy"],
        width,
        label="Original (SAE)",
        color="#2ecc71",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        benchmark_stats["transformed_accuracy"],
        width,
        label="Transformed (Dialect)",
        color="#3498db",
        edgecolor="black",
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Average Accuracy by Benchmark (Across All Dialects)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_stats["benchmark"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def _format_dialect_name(dialect: str) -> str:
    """Format dialect name for display."""
    return dialect.replace("_", " ").title()


def _format_benchmark_name(benchmark: str) -> str:
    """Format benchmark name for display."""
    name_map = {
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
        "arc_challenge": "ARC Challenge",
        "hellaswag": "HellaSwag",
        "realtoxicityprompts": "RealToxicity",
    }
    return name_map.get(benchmark, benchmark.replace("_", " ").title())


def generate_report(
    audits: list[AuditData],
    output_dir: Path,
    title: str = "Dialectal Fairness Audit Report",
) -> Path:
    """
    Generate an HTML report with all visualizations.

    Args:
        audits: List of audit results
        output_dir: Directory to save report and images
        title: Report title

    Returns:
        Path to generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Close any existing figures
    plt.close("all")

    # Generate all visualizations
    heatmap_path = output_dir / "accuracy_heatmap.png"
    gap_path = output_dir / "accuracy_gap.png"
    dialect_path = output_dir / "dialect_comparison.png"
    benchmark_path = output_dir / "benchmark_comparison.png"

    create_accuracy_heatmap(audits, heatmap_path)
    create_accuracy_gap_chart(audits, gap_path)
    create_dialect_comparison(audits, dialect_path)
    create_benchmark_comparison(audits, benchmark_path)

    plt.close("all")

    # Build summary statistics
    df = _audits_to_dataframe(audits)

    overall_orig = df["original_accuracy"].mean()
    overall_trans = df["transformed_accuracy"].mean()
    overall_gap = df["accuracy_gap"].mean()
    total_pairs = df["total_pairs"].sum()

    # Worst performing combinations
    df_sorted = df.sort_values("accuracy_gap")
    worst_gaps = df_sorted.head(5)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary-box {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .positive {{
            color: #27ae60;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .findings {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="summary-box">
        <h2>Executive Summary</h2>
        <p>This report analyzes the fairness of <strong>{df['model'].iloc[0]}</strong> across
        <strong>{len(df['dialect'].unique())}</strong> dialects and
        <strong>{len(df['benchmark'].unique())}</strong> benchmarks,
        covering <strong>{total_pairs:,}</strong> total evaluation pairs.</p>
    </div>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{overall_orig:.1%}</div>
            <div class="stat-label">Original (SAE) Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{overall_trans:.1%}</div>
            <div class="stat-label">Transformed (Dialect) Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'negative' if overall_gap < 0 else 'positive'}">{overall_gap:+.1%}</div>
            <div class="stat-label">Average Accuracy Gap</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_pairs:,}</div>
            <div class="stat-label">Total Evaluation Pairs</div>
        </div>
    </div>

    <div class="findings">
        <h3>Key Findings</h3>
        <ul>
            <li>The model shows a <strong>{abs(overall_gap):.1%}</strong> accuracy degradation when processing dialect-transformed text.</li>
            <li>This represents potential bias against non-standard English speakers.</li>
            <li>Largest gaps observed in: {', '.join(worst_gaps['benchmark'].head(3).tolist())}</li>
        </ul>
    </div>

    <div class="chart-container">
        <h2>Accuracy Heatmaps</h2>
        <p>Side-by-side comparison of accuracy on original (SAE) vs dialect-transformed prompts.</p>
        <img src="accuracy_heatmap.png" alt="Accuracy Heatmap">
    </div>

    <div class="chart-container">
        <h2>Accuracy Gap</h2>
        <p>Difference between dialect and SAE accuracy. Negative values (red) indicate the model performs worse on dialect text.</p>
        <img src="accuracy_gap.png" alt="Accuracy Gap Chart">
    </div>

    <div class="chart-container">
        <h2>Dialect Comparison</h2>
        <p>Average accuracy across all benchmarks, grouped by dialect.</p>
        <img src="dialect_comparison.png" alt="Dialect Comparison">
    </div>

    <div class="chart-container">
        <h2>Benchmark Comparison</h2>
        <p>Average accuracy across all dialects, grouped by benchmark.</p>
        <img src="benchmark_comparison.png" alt="Benchmark Comparison">
    </div>

    <h2>Detailed Results</h2>

    <h3>Largest Accuracy Gaps</h3>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Dialect</th>
            <th>Original Acc.</th>
            <th>Dialect Acc.</th>
            <th>Gap</th>
            <th>Pairs</th>
        </tr>
"""

    for _, row in worst_gaps.iterrows():
        html += f"""        <tr>
            <td>{_format_benchmark_name(row['benchmark'])}</td>
            <td>{_format_dialect_name(row['dialect'])}</td>
            <td>{row['original_accuracy']:.1%}</td>
            <td>{row['transformed_accuracy']:.1%}</td>
            <td class="negative">{row['accuracy_gap']:+.1%}</td>
            <td>{row['total_pairs']}</td>
        </tr>
"""

    html += """    </table>

    <h3>All Results</h3>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Dialect</th>
            <th>Original Acc.</th>
            <th>Dialect Acc.</th>
            <th>Gap</th>
            <th>Pairs</th>
        </tr>
"""

    for _, row in df.sort_values(["benchmark", "dialect"]).iterrows():
        gap_class = "negative" if row["accuracy_gap"] < 0 else "positive"
        html += f"""        <tr>
            <td>{_format_benchmark_name(row['benchmark'])}</td>
            <td>{_format_dialect_name(row['dialect'])}</td>
            <td>{row['original_accuracy']:.1%}</td>
            <td>{row['transformed_accuracy']:.1%}</td>
            <td class="{gap_class}">{row['accuracy_gap']:+.1%}</td>
            <td>{row['total_pairs']}</td>
        </tr>
"""

    html += """    </table>

    <div class="summary-box">
        <h3>Methodology</h3>
        <p>Each benchmark question was transformed into dialect variants using LLM-based
        transformation with post-processing validation. The same model was then evaluated
        on both the original Standard American English (SAE) prompts and their dialect-transformed
        versions. Accuracy gaps represent the difference in model performance between dialect
        and SAE text.</p>
    </div>

    <footer style="text-align: center; color: #7f8c8d; margin-top: 40px; padding: 20px;">
        <p>Generated by Dialectal Fairness Auditing Pipeline</p>
    </footer>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html)

    print(f"Report generated: {report_path}")
    return report_path
