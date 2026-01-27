#!/usr/bin/env python
"""Generate dialect transformations for benchmark datasets.

Supports multi-machine parallelization via --start and --end index parameters.

Example usage:
    # Generate full dataset
    python run_generation.py --benchmark gsm8k --dialect hiberno_english

    # Split across 3 machines
    # Machine 1:
    python run_generation.py --benchmark gsm8k --dialect hiberno_english --start 0 --end 500
    # Machine 2:
    python run_generation.py --benchmark gsm8k --dialect hiberno_english --start 500 --end 1000
    # Machine 3:
    python run_generation.py --benchmark gsm8k --dialect hiberno_english --start 1000 --end 1319

    # Combine results
    python combine_results.py generation data/benchmarks/gsm8k_*_0-500.json data/benchmarks/gsm8k_*_500-1000.json ...
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import List

from src.benchmarks import load_benchmark, get_available_benchmarks, BenchmarkSample
from src.generation import DatasetGenerator, GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dialect-transformed benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        choices=get_available_benchmarks(),
        help="Benchmark to transform",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="hiberno_english",
        help="Target dialect",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="Start index (inclusive) for slicing samples",
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End index (exclusive) for slicing samples",
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
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
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


class SlicedBenchmark:
    """Wrapper to slice a benchmark dataset by index range."""

    def __init__(self, samples: List[BenchmarkSample], name: str):
        self._samples = samples
        self.name = name

    @property
    def samples(self):
        return iter(self._samples)


def main():
    args = parse_args()

    print(f"Loading {args.benchmark} dataset ({args.split} split)...", flush=True)
    benchmark = load_benchmark(args.benchmark, split=args.split)
    samples = list(benchmark.samples)
    total_samples = len(samples)
    print(f"Loaded {total_samples} samples", flush=True)

    # Apply slicing
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else total_samples
    samples = samples[start:end]
    print(f"Processing samples [{start}:{end}] ({len(samples)} samples)")

    # Create sliced benchmark wrapper
    sliced_benchmark = SlicedBenchmark(samples, benchmark.name)

    # Configure generator
    config = GenerationConfig(
        dialect=args.dialect,
        backend=args.backend,
        model=args.model,
        output_dir=str(args.output_dir),
    )
    generator = DatasetGenerator(config=config)

    # Generate
    result = generator.generate(sliced_benchmark, show_progress=not args.quiet)

    print(f"\n=== Generation Complete ===")
    print(f"Pass rate: {result.pass_rate:.1%}")
    print(f"Valid pairs: {result.valid_pairs}/{result.total_samples}")
    print(f"Transform errors: {result.transform_errors}")
    print(f"Failed validation: {result.failed_validation}")

    # Save results
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include index range in filename for easy identification
        index_suffix = f"_{start}-{end}" if args.start is not None or args.end is not None else ""
        filename = f"{args.benchmark}_{args.dialect}{index_suffix}_{timestamp}.json"
        output_path = generator.save_result(result, filename)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
