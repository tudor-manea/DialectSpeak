#!/bin/bash
# Generation split for Mac Mini (M4 Pro, 24GB)
# 15 combinations: all 9 benchmarks for hiberno_english + 6 for aave
set -e

cd ~/Projects/Capstone
source venv/bin/activate
export PYTHONPATH=~/Projects/Capstone

MODEL="${MODEL:-qwen2.5:14b}"
MAX=300
echo "Mac Mini: $MODEL, $MAX samples per combo"

run() {
    local bench=$1 dialect=$2 split=$3
    echo "=== $bench x $dialect ==="
    python3 scripts/run_generation.py \
        --benchmark "$bench" --dialect "$dialect" \
        --backend ollama --model "$MODEL" \
        --split "$split" --end "$MAX"
}

# All hiberno_english (9 combos)
run gsm8k hiberno_english test
run mmlu hiberno_english test
run arc hiberno_english test
run hellaswag hiberno_english validation
run boolq hiberno_english validation
run truthfulqa hiberno_english validation
run realtoxicityprompts hiberno_english train
run donotanswer hiberno_english train
run toxigen hiberno_english train

# AAVE (6 combos)
run gsm8k aave test
run mmlu aave test
run arc aave test
run hellaswag aave validation
run boolq aave validation
run truthfulqa aave validation

echo "=== Mac Mini generation complete ==="
