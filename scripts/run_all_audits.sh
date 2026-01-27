#!/bin/bash
# Run fairness audits on all 13 dialect pair datasets
# Requires: venv activated, Ollama running with llama3.1:8b

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source venv/bin/activate

DATASETS=(
    "data/benchmarks/arc_hiberno_english_20260125_145323.json"
    "data/benchmarks/arc_aave_20260125_151805.json"
    "data/benchmarks/arc_indian_english_20260125_154413.json"
    "data/benchmarks/hellaswag_hiberno_english_0-1000_20260125_180440.json"
    "data/benchmarks/hellaswag_aave_0-1000_20260125_182110.json"
    "data/benchmarks/hellaswag_indian_english_0-1000_20260125_183605.json"
    "data/benchmarks/mmlu_hiberno_english_0-1000_20260125_161118.json"
    "data/benchmarks/mmlu_aave_0-1000_20260125_163247.json"
    "data/benchmarks/mmlu_indian_english_0-1000_20260125_165641.json"
    "data/benchmarks/realtoxicityprompts_hiberno_english_0-1000_20260125_185421.json"
    "data/benchmarks/realtoxicityprompts_aave_0-1000_20260125_191103.json"
    "data/benchmarks/realtoxicityprompts_indian_english_0-1000_20260125_193222.json"
    "data/benchmarks/gsm8k_hiberno_english_20260121_002828.json"
)

TOTAL=${#DATASETS[@]}
COMPLETED=0
START_TIME=$(date +%s)

mkdir -p data/audits

for dataset in "${DATASETS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    NAME=$(basename "$dataset" .json)

    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $NAME"
    echo "=============================================="

    if [[ ! -f "$dataset" ]]; then
        echo "WARNING: File not found, skipping: $dataset"
        continue
    fi

    python scripts/run_audit.py --pairs "$dataset" --backend ollama --model llama3.1:8b

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "Elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=============================================="
echo "COMPLETE - Total: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo "Results: data/audits/"
echo "=============================================="
