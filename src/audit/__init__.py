"""
Audit Module

Fairness auditing by comparing LLM responses on original vs dialect-transformed prompts.
"""

from .evaluator import (
    AuditConfig,
    AuditResult,
    FairnessAuditor,
    PairResult,
    run_audit,
    load_generated_pairs,
    get_benchmark_type,
    BENCHMARK_TYPES,
    CHOICE_LABELS,
)

__all__ = [
    "AuditConfig",
    "AuditResult",
    "FairnessAuditor",
    "PairResult",
    "run_audit",
    "load_generated_pairs",
    "get_benchmark_type",
    "BENCHMARK_TYPES",
    "CHOICE_LABELS",
]
