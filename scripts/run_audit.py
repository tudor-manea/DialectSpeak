#!/usr/bin/env python3
"""Run fairness audit on generated dialect pairs.

Supports multi-machine parallelization via --start and --end index parameters.

Example usage:
    # Run on full dataset
    python run_audit.py --pairs data/benchmarks/gsm8k_hiberno_english_20260121_002828.json

    # Split across 3 machines
    # Machine 1:
    python run_audit.py --pairs data/benchmarks/gsm8k_hiberno_english_20260121_002828.json --start 0 --end 250
    # Machine 2:
    python run_audit.py --pairs data/benchmarks/gsm8k_hiberno_english_20260121_002828.json --start 250 --end 500
    # Machine 3:
    python run_audit.py --pairs data/benchmarks/gsm8k_hiberno_english_20260121_002828.json --start 500 --end 721

    # Combine results
    python combine_results.py audit data/audits/audit_*_0-250.json data/audits/audit_*_250-500.json ...
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.audit import load_generated_pairs, FairnessAuditor, AuditConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fairness audit on dialect-transformed pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pairs", "-p",
        type=Path,
        default=Path("data/benchmarks/gsm8k_hiberno_english_20260121_002828.json"),
        help="Path to generated pairs JSON file",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="Start index (inclusive) for slicing pairs",
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End index (exclusive) for slicing pairs",
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="ollama",
        choices=["ollama", "openai"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.1:8b",
        help="Model name",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audits"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bar",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load pairs
    pairs, benchmark, dialect = load_generated_pairs(args.pairs)
    total_pairs = len(pairs)
    print(f"Loaded {total_pairs} pairs from {benchmark} ({dialect})")

    # Apply slicing
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else total_pairs
    pairs = pairs[start:end]
    print(f"Processing pairs [{start}:{end}] ({len(pairs)} pairs)")

    # Run audit
    config = AuditConfig(backend=args.backend, model=args.model)
    auditor = FairnessAuditor(config)

    result = auditor.audit(pairs, benchmark, dialect, show_progress=not args.quiet)

    # Print results
    print("\n" + result.summary())

    # Show mismatches
    mismatches = [p for p in result.pairs if p.original_correct != p.transformed_correct and not p.error]
    if mismatches:
        print(f"\n\n=== Cases where dialect affected accuracy ({len(mismatches)} total) ===")
        for p in mismatches[:10]:  # Show first 10
            print(f"\n--- {p.id} ---")
            print(f"Expected: {p.expected_answer}")
            print(f"Original: {p.original_answer} ({'✓' if p.original_correct else '✗'})")
            print(f"Transformed: {p.transformed_answer} ({'✓' if p.transformed_correct else '✗'})")
        if len(mismatches) > 10:
            print(f"\n... and {len(mismatches) - 10} more mismatches")

    # Save results
    if not args.no_save:
        if args.output:
            output_path = args.output
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_safe = args.model.replace(":", "_").replace("/", "_")
            # Include index range in filename for easy identification
            index_suffix = f"_{start}-{end}" if args.start is not None or args.end is not None else ""
            filename = f"audit_{benchmark}_{dialect}_{model_safe}{index_suffix}_{timestamp}.json"
            output_path = args.output_dir / filename

        auditor.save_result(result, output_path)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
