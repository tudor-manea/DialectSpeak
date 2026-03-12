#!/bin/bash
# Laptop audit script: Indian English (9) = 9 audits
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

MODEL="${MODEL:-qwen2.5:14b}"
echo "Using model: $MODEL"

DATASETS=(
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

TOTAL=${#DATASETS[@]}
COMPLETED=0
START_TIME=$(date +%s)

mkdir -p data/audits

echo "=============================================="
echo "Laptop: $TOTAL audits to run"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    NAME=$(basename "$dataset" .json)

    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $NAME"
    echo "=============================================="

    EXISTING=$(ls data/audits/audit_${NAME}_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$EXISTING" ]]; then
        echo "SKIP: Audit already exists: $EXISTING"
        echo ""
        continue
    fi

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
echo "LAPTOP AUDITS COMPLETE"
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo "=============================================="
