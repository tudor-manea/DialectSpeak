#!/usr/bin/env python
"""Generate Hiberno-English transformations for full GSM8K dataset."""

from src.benchmarks import load_gsm8k
from src.generation import generate_dataset

print("Loading GSM8K dataset...", flush=True)
gsm8k = load_gsm8k(split="test")
print(f"Loaded {len(list(gsm8k.samples))} samples", flush=True)

result = generate_dataset(
    benchmark=gsm8k,
    dialect="hiberno_english",
    backend="ollama",
    model="llama3.1:8b",
    output_dir="data/benchmarks",
    save=True,
)

print(f"\n=== Generation Complete ===")
print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Valid pairs: {result.valid_pairs}")
