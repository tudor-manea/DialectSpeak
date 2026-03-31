"""
Analysis Module

Visualization and reporting for fairness audit results.
"""

from .visualize import (
    load_audit,
    load_all_audits,
    create_accuracy_heatmap,
    create_dialect_comparison,
    create_accuracy_gap_chart,
    generate_report,
    AuditData,
)
from .presentation import generate_all as generate_presentation

__all__ = [
    "load_audit",
    "load_all_audits",
    "create_accuracy_heatmap",
    "create_dialect_comparison",
    "create_accuracy_gap_chart",
    "generate_report",
    "generate_presentation",
    "AuditData",
]
