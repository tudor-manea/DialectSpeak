#!/usr/bin/env python3
"""Run fairness audit on generated dialect pairs."""

from pathlib import Path
from src.audit import load_generated_pairs, FairnessAuditor, AuditConfig

# Configuration
PAIRS_PATH = Path("data/benchmarks/gsm8k_hiberno_english_20260121_002828.json")
NUM_PAIRS = 15  # Adjust as needed

# Load pairs
pairs, benchmark, dialect = load_generated_pairs(PAIRS_PATH)
print(f"Loaded {len(pairs)} pairs from {benchmark} ({dialect})")

# Run audit
config = AuditConfig(backend="ollama", model="llama3.1:8b")
auditor = FairnessAuditor(config)

print(f"\nAuditing {NUM_PAIRS} pairs...")
result = auditor.audit(pairs[:NUM_PAIRS], benchmark, dialect, show_progress=True)

# Print results
print("\n" + result.summary())

# Show mismatches
mismatches = [p for p in result.pairs if p.original_correct != p.transformed_correct and not p.error]
if mismatches:
    print(f"\n\n=== Cases where dialect affected accuracy ({len(mismatches)} total) ===")
    for p in mismatches:
        print(f"\n--- {p.id} ---")
        print(f"Expected: {p.expected_answer}")
        print(f"Original: {p.original_answer} ({'✓' if p.original_correct else '✗'})")
        print(f"Transformed: {p.transformed_answer} ({'✓' if p.transformed_correct else '✗'})")

# Save results
output_path = Path("data/audits/audit_15_sample.json")
auditor.save_result(result, output_path)
print(f"\nSaved to: {output_path}")
