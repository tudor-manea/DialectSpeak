#!/bin/bash
# Mac Mini audit script: Hiberno-English (9) + AAVE (9) = 18 audits
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

MODEL="${MODEL:-qwen2.5:14b}"
echo "Using model: $MODEL"

DATASETS=(
    data/benchmarks/gsm8k_hiberno_english_0-300_20260301_193657.json
    data/benchmarks/mmlu_hiberno_english_0-300_20260301_194748.json
    data/benchmarks/arc_hiberno_english_0-300_20260301_195955.json
    data/benchmarks/hellaswag_hiberno_english_0-300_20260301_201241.json
    data/benchmarks/boolq_hiberno_english_0-300_20260301_205235.json
    data/benchmarks/truthfulqa_hiberno_english_0-300_20260301_205844.json
    data/benchmarks/realtoxicityprompts_hiberno_english_0-300_20260301_210829.json
    data/benchmarks/donotanswer_hiberno_english_0-300_20260301_211509.json
    data/benchmarks/toxigen_hiberno_english_0-300_20260301_212923.json
    data/benchmarks/gsm8k_aave_0-300_20260301_221903.json
    data/benchmarks/mmlu_aave_0-300_20260301_231254.json
    data/benchmarks/arc_aave_0-300_20260301_235243.json
    data/benchmarks/hellaswag_aave_0-300_20260302_002727.json
    data/benchmarks/boolq_aave_0-300_20260302_013047.json
    data/benchmarks/truthfulqa_aave_0-300_20260302_023004.json
    data/benchmarks/realtoxicityprompts_aave_0-300_20260302_095246.json
    data/benchmarks/donotanswer_aave_0-300_20260302_150916.json
    data/benchmarks/toxigen_aave_0-300_20260302_160758.json
)

TOTAL=${#DATASETS[@]}
COMPLETED=0
START_TIME=$(date +%s)

mkdir -p data/audits

echo "=============================================="
echo "Mac Mini: $TOTAL audits to run"
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
echo "MAC MINI AUDITS COMPLETE"
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo "=============================================="
