#!/usr/bin/env python3
"""Combine multiple audit or generation result files into a single file.

Use this after running parallel jobs across multiple machines.

Example usage:
    # Combine audit results
    python combine_results.py audit data/audits/audit_*_0-250.json data/audits/audit_*_250-500.json -o combined_audit.json

    # Combine generation results
    python combine_results.py generation data/benchmarks/gsm8k_*_0-500.json data/benchmarks/gsm8k_*_500-1000.json -o combined_generation.json

    # Using glob patterns (shell expansion)
    python combine_results.py audit data/audits/audit_gsm8k_hiberno_english_*.json -o combined.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def combine_audit_results(files: List[Path]) -> Dict[str, Any]:
    """
    Combine multiple audit result files.

    Merges pair results and recalculates aggregate statistics.
    """
    if not files:
        raise ValueError("No files to combine")

    # Load all results
    all_results = []
    for f in files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    # Verify compatibility
    first = all_results[0]
    for r in all_results[1:]:
        if r["benchmark"] != first["benchmark"]:
            raise ValueError(f"Benchmark mismatch: {r['benchmark']} vs {first['benchmark']}")
        if r["dialect"] != first["dialect"]:
            raise ValueError(f"Dialect mismatch: {r['dialect']} vs {first['dialect']}")
        if r["model"] != first["model"]:
            raise ValueError(f"Model mismatch: {r['model']} vs {first['model']}")

    # Merge pairs (deduplicate by ID)
    seen_ids = set()
    merged_pairs = []
    for r in all_results:
        for pair in r["pairs"]:
            if pair["id"] not in seen_ids:
                seen_ids.add(pair["id"])
                merged_pairs.append(pair)

    # Sort by ID for consistent ordering
    merged_pairs.sort(key=lambda p: p["id"])

    # Recalculate statistics
    original_correct = 0
    transformed_correct = 0
    both_correct = 0
    both_wrong = 0
    original_only = 0
    transformed_only = 0
    errors = 0

    for pair in merged_pairs:
        if pair.get("error"):
            errors += 1
            continue

        orig_correct = pair.get("original_correct", False)
        trans_correct = pair.get("transformed_correct", False)

        if orig_correct:
            original_correct += 1
        if trans_correct:
            transformed_correct += 1

        if orig_correct and trans_correct:
            both_correct += 1
        elif not orig_correct and not trans_correct:
            both_wrong += 1
        elif orig_correct:
            original_only += 1
        else:
            transformed_only += 1

    total = len(merged_pairs) - errors
    orig_acc = original_correct / total if total else 0.0
    trans_acc = transformed_correct / total if total else 0.0

    return {
        "benchmark": first["benchmark"],
        "dialect": first["dialect"],
        "model": first["model"],
        "total_pairs": total,
        "original_correct": original_correct,
        "transformed_correct": transformed_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "original_only_correct": original_only,
        "transformed_only_correct": transformed_only,
        "original_accuracy": orig_acc,
        "transformed_accuracy": trans_acc,
        "accuracy_gap": trans_acc - orig_acc,
        "pairs": merged_pairs,
        "audit_time": datetime.now().isoformat(),
        "errors": errors,
        "combined_from": [str(f) for f in files],
    }


def combine_generation_results(files: List[Path]) -> Dict[str, Any]:
    """
    Combine multiple generation result files.

    Merges pairs and recalculates aggregate statistics.
    """
    if not files:
        raise ValueError("No files to combine")

    # Load all results
    all_results = []
    for f in files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    # Verify compatibility
    first = all_results[0]
    for r in all_results[1:]:
        if r["benchmark"] != first["benchmark"]:
            raise ValueError(f"Benchmark mismatch: {r['benchmark']} vs {first['benchmark']}")
        if r["dialect"] != first["dialect"]:
            raise ValueError(f"Dialect mismatch: {r['dialect']} vs {first['dialect']}")

    # Merge pairs (deduplicate by ID)
    seen_ids = set()
    merged_pairs = []
    for r in all_results:
        for pair in r["pairs"]:
            if pair["id"] not in seen_ids:
                seen_ids.add(pair["id"])
                merged_pairs.append(pair)

    # Sort by ID for consistent ordering
    merged_pairs.sort(key=lambda p: p["id"])

    # Recalculate statistics
    total_samples = sum(r["total_samples"] for r in all_results)
    successful_transforms = sum(r["successful_transforms"] for r in all_results)
    valid_pairs = len(merged_pairs)
    failed_validation = sum(r["failed_validation"] for r in all_results)
    transform_errors = sum(r["transform_errors"] for r in all_results)
    pass_rate = valid_pairs / total_samples if total_samples else 0.0

    return {
        "benchmark": first["benchmark"],
        "dialect": first["dialect"],
        "total_samples": total_samples,
        "successful_transforms": successful_transforms,
        "valid_pairs": valid_pairs,
        "failed_validation": failed_validation,
        "transform_errors": transform_errors,
        "pass_rate": pass_rate,
        "pairs": merged_pairs,
        "generation_time": datetime.now().isoformat(),
        "combined_from": [str(f) for f in files],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple result files from parallel runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["audit", "generation"],
        help="Type of results to combine",
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Result files to combine",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify all files exist
    for f in args.files:
        if not f.exists():
            raise FileNotFoundError(f"File not found: {f}")

    print(f"Combining {len(args.files)} {args.mode} result files...")

    if args.mode == "audit":
        combined = combine_audit_results(args.files)
        print(f"Combined {combined['total_pairs']} pairs")
        print(f"Original accuracy: {combined['original_accuracy']:.1%}")
        print(f"Transformed accuracy: {combined['transformed_accuracy']:.1%}")
        print(f"Accuracy gap: {combined['accuracy_gap']:+.1%}")
    else:
        combined = combine_generation_results(args.files)
        print(f"Combined {combined['valid_pairs']} valid pairs from {combined['total_samples']} samples")
        print(f"Pass rate: {combined['pass_rate']:.1%}")

    # Save combined results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
