"""
Statistical significance testing for fairness audits.

Implements McNemar's test for paired nominal data:
given that each prompt is evaluated in both SAE and dialect form,
McNemar's test determines whether the proportion of correct answers
differs significantly between the two conditions.

Includes Benjamini-Hochberg FDR correction for multiple comparisons
and bootstrap confidence intervals on accuracy gaps.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2


@dataclass
class McNemarResult:
    """Result of McNemar's test on a single audit."""

    benchmark: str
    dialect: str
    model: str
    # Contingency table
    both_correct: int
    both_wrong: int
    original_only: int       # SAE correct, dialect wrong (b)
    transformed_only: int    # dialect correct, SAE wrong (c)
    # Test results
    statistic: float | None
    p_value: float
    significant_01: bool     # p < 0.01
    significant_05: bool     # p < 0.05
    method: str              # "exact" or "chi2"
    # BH-corrected results (set by apply_bh_correction)
    bh_significant_05: bool = False
    bh_significant_01: bool = False
    adjusted_p: float | None = None
    # Bootstrap CI on accuracy gap (set by add_bootstrap_ci)
    accuracy_gap: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


def mcnemar_test(
    original_only: int,
    transformed_only: int,
    exact_threshold: int = 25,
    continuity: bool = True,
) -> tuple[float | None, float, str]:
    """
    Run McNemar's test on discordant pair counts.

    Args:
        original_only: count where SAE correct but dialect wrong (b)
        transformed_only: count where dialect correct but SAE wrong (c)
        exact_threshold: use exact binomial test when b + c < this
        continuity: apply continuity correction for chi-squared

    Returns:
        (statistic, p_value, method)
    """
    b, c = original_only, transformed_only
    n = b + c

    if n == 0:
        return None, 1.0, "exact"

    if n < exact_threshold:
        # Exact binomial test: H0 is that b ~ Binomial(n, 0.5)
        p_value = binomtest(b, n, 0.5).pvalue
        return None, p_value, "exact"

    # Chi-squared approximation
    if continuity:
        stat = (abs(b - c) - 1) ** 2 / n
    else:
        stat = (b - c) ** 2 / n

    p_value = 1 - chi2.cdf(stat, df=1)
    return stat, p_value, "chi2"


def test_audit(audit) -> McNemarResult:
    """Run McNemar's test on a single AuditData object."""
    b = audit.original_only_correct
    c = audit.transformed_only_correct

    stat, p_val, method = mcnemar_test(b, c)

    return McNemarResult(
        benchmark=audit.benchmark,
        dialect=audit.dialect,
        model=audit.model,
        both_correct=audit.both_correct,
        both_wrong=audit.both_wrong,
        original_only=b,
        transformed_only=c,
        statistic=stat,
        p_value=p_val,
        significant_01=p_val < 0.01,
        significant_05=p_val < 0.05,
        method=method,
    )


def test_all_audits(audits: list) -> list[McNemarResult]:
    """Run McNemar's test on all audits, return sorted by p-value."""
    results = [test_audit(a) for a in audits]
    results.sort(key=lambda r: r.p_value)
    return results


def apply_bh_correction(results: list[McNemarResult], alpha: float = 0.05) -> list[McNemarResult]:
    """Apply Benjamini-Hochberg FDR correction to a list of McNemar results.

    Modifies results in-place, setting bh_significant_05, bh_significant_01,
    and adjusted_p fields.
    """
    sorted_results = sorted(results, key=lambda r: r.p_value)
    m = len(sorted_results)

    # Compute adjusted p-values (step-up procedure)
    adjusted = [0.0] * m
    for i in range(m - 1, -1, -1):
        rank = i + 1
        raw = sorted_results[i].p_value * m / rank
        if i == m - 1:
            adjusted[i] = min(raw, 1.0)
        else:
            adjusted[i] = min(raw, adjusted[i + 1])

    for i, r in enumerate(sorted_results):
        r.adjusted_p = adjusted[i]
        r.bh_significant_05 = adjusted[i] < 0.05
        r.bh_significant_01 = adjusted[i] < 0.01

    return results


def bootstrap_accuracy_gap(
    original_correct: list[bool],
    transformed_correct: list[bool],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval on accuracy gap.

    Returns (gap, ci_lower, ci_upper) where gap = transformed_acc - original_acc.
    """
    rng = np.random.default_rng(seed)
    orig = np.array(original_correct, dtype=float)
    trans = np.array(transformed_correct, dtype=float)
    n = len(orig)

    observed_gap = trans.mean() - orig.mean()

    gaps = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gaps[i] = trans[idx].mean() - orig[idx].mean()

    alpha = 1 - confidence
    ci_lower = float(np.percentile(gaps, 100 * alpha / 2))
    ci_upper = float(np.percentile(gaps, 100 * (1 - alpha / 2)))

    return float(observed_gap), ci_lower, ci_upper


def add_bootstrap_ci(
    result: McNemarResult,
    audit_path: Path,
    n_bootstrap: int = 10000,
) -> McNemarResult:
    """Load pair-level data from audit JSON and add bootstrap CI to result."""
    with open(audit_path) as f:
        data = json.load(f)

    # Filter out pairs where correctness is None (e.g. parsing failures)
    pairs = [p for p in data["pairs"]
             if p["original_correct"] is not None and p["transformed_correct"] is not None]
    orig = [p["original_correct"] for p in pairs]
    trans = [p["transformed_correct"] for p in pairs]

    gap, ci_lo, ci_hi = bootstrap_accuracy_gap(orig, trans, n_bootstrap=n_bootstrap)
    result.accuracy_gap = gap
    result.ci_lower = ci_lo
    result.ci_upper = ci_hi
    return result
