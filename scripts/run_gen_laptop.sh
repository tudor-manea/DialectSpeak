#!/bin/bash
# Generation split for MacBook Air (M3, 16GB)
# 12 combinations: remaining AAVE (3 toxicity) + all indian_english (9)
cd ~/Projects/Capstone
source venv313/bin/activate
export PYTHONPATH=~/Projects/Capstone

MODEL="${MODEL:-qwen2.5:14b}"
MAX=300
echo "Laptop: $MODEL, $MAX samples per combo"

run() {
    local bench=$1 dialect=$2 split=$3
    echo "=== $bench x $dialect ==="

    # Skip if already generated
    if ls data/benchmarks/${bench}_${dialect}_*.json 2>/dev/null | grep -q .; then
        echo "SKIP: already exists"
        return
    fi

    python3 scripts/run_generation.py \
        --benchmark "$bench" --dialect "$dialect" \
        --backend ollama --model "$MODEL" \
        --split "$split" --end "$MAX" || echo "ERROR: $bench x $dialect failed, continuing..."
}

# Remaining AAVE (3 toxicity combos)
run realtoxicityprompts aave train
run donotanswer aave train
run toxigen aave train

# All indian_english (9 combos)
run gsm8k indian_english test
run mmlu indian_english test
run arc indian_english test
run hellaswag indian_english validation
run boolq indian_english validation
run truthfulqa indian_english validation
run realtoxicityprompts indian_english train
run donotanswer indian_english train
run toxigen indian_english train

echo "=== Laptop generation complete ==="
