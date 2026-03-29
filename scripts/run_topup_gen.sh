#!/bin/bash
# Top-up generation for benchmarks with <200 valid pairs.
# Generates from indices 300-600 (fresh prompts not seen before).
# Run on Mac Mini overnight.
cd ~/Projects/Capstone
source venv/bin/activate
export PYTHONPATH=~/Projects/Capstone

MODEL="${MODEL:-qwen2.5:14b}"
echo "Top-up generation: $MODEL, indices 300-600"

run() {
    local bench=$1 dialect=$2 split=$3
    echo "=== $bench x $dialect (top-up 300-600) ==="
    python3 scripts/run_generation.py \
        --benchmark "$bench" --dialect "$dialect" \
        --backend ollama --model "$MODEL" \
        --split "$split" --start 300 --end 600 || echo "ERROR: $bench x $dialect failed, continuing..."
}

# AAVE top-ups (6 runs, all <200 pairs)
run arc_challenge   aave test
run mmlu            aave test
run boolq           aave validation
run realtoxicityprompts aave train
run donotanswer     aave train
run truthfulqa      aave validation

# Hiberno-English top-ups (2 runs, <150 pairs)
run donotanswer     hiberno_english train
run truthfulqa      hiberno_english validation

echo "=== Top-up generation complete ==="
echo "Next: combine results with combine_results.py"
