#!/bin/bash
# Gap-fill script: missing Indian English audits for llama3.1:8b (9) and qwen2.5:14b (4)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

TOTAL=13
COMPLETED=0
START_TIME=$(date +%s)

echo "=============================================="
echo "Gap-fill: $TOTAL audits to run"
echo "=============================================="
echo ""

# --- llama3.1:8b: all 9 Indian English ---

MODEL="llama3.1:8b"
echo "Pulling $MODEL..."
ollama pull "$MODEL"

LLAMA_DATASETS=(
    data/benchmarks/gsm8k_indian_english_0-300_20260303_091722.json
    data/benchmarks/mmlu_indian_english_0-300_20260303_094652.json
    data/benchmarks/arc_indian_english_0-300_20260303_102245.json
    data/benchmarks/hellaswag_indian_english_0-300_20260303_110000.json
    data/benchmarks/boolq_indian_english_0-300_20260304_010831.json
    data/benchmarks/truthfulqa_indian_english_0-300_20260304_012840.json
    data/benchmarks/realtoxicityprompts_indian_english_0-300_20260304_020748.json
    data/benchmarks/donotanswer_indian_english_0-300_20260304_023449.json
    data/benchmarks/toxigen_indian_english_0-300_20260304_032719.json
)

for dataset in "${LLAMA_DATASETS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    NAME=$(basename "$dataset" .json)
    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $MODEL — $NAME"
    echo "=============================================="

    python3 scripts/run_audit.py --pairs "$dataset" --backend ollama --model "$MODEL"

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    AVG=$((ELAPSED / COMPLETED))
    REMAINING=$(( (TOTAL - COMPLETED) * AVG ))
    echo "Elapsed: $((ELAPSED / 60))m | ETA: $((REMAINING / 60))m"
    echo ""
done

# --- qwen2.5:14b: 4 missing Indian English ---

MODEL="qwen2.5:14b"
echo "Pulling $MODEL..."
ollama pull "$MODEL"

QWEN_DATASETS=(
    data/benchmarks/boolq_indian_english_0-300_20260304_010831.json
    data/benchmarks/truthfulqa_indian_english_0-300_20260304_012840.json
    data/benchmarks/donotanswer_indian_english_0-300_20260304_023449.json
    data/benchmarks/toxigen_indian_english_0-300_20260304_032719.json
)

for dataset in "${QWEN_DATASETS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    NAME=$(basename "$dataset" .json)
    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $MODEL — $NAME"
    echo "=============================================="

    python3 scripts/run_audit.py --pairs "$dataset" --backend ollama --model "$MODEL"

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    AVG=$((ELAPSED / COMPLETED))
    REMAINING=$(( (TOTAL - COMPLETED) * AVG ))
    echo "Elapsed: $((ELAPSED / 60))m | ETA: $((REMAINING / 60))m"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=============================================="
echo "GAP-FILL COMPLETE"
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo "=============================================="
