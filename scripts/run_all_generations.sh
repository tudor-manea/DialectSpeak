#!/bin/bash
# Generate dialect transformations for all benchmark x dialect combinations
# Requires: venv activated, Ollama running with the target model
# Default model: gemma3:27b (override with MODEL env var)
# Default max samples per benchmark: 1000 (override with MAX_SAMPLES env var)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

MODEL="${MODEL:-gemma3:27b}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
echo "Using model: $MODEL"
echo "Max samples per benchmark: $MAX_SAMPLES"

DIALECTS=("hiberno_english" "aave" "indian_english")

# Benchmark configs: name|split|extra_args
BENCHMARKS=(
    "gsm8k|test|"
    "mmlu|test|"
    "arc|test|"
    "hellaswag|validation|"
    "boolq|validation|"
    "truthfulqa|validation|"
    "realtoxicityprompts|train|"
    "donotanswer|train|"
    "toxigen|train|"
)

# Count total combinations
TOTAL=$((${#BENCHMARKS[@]} * ${#DIALECTS[@]}))
COMPLETED=0
START_TIME=$(date +%s)

mkdir -p data/benchmarks

echo "=============================================="
echo "Generating $TOTAL combinations (${#BENCHMARKS[@]} benchmarks x ${#DIALECTS[@]} dialects)"
echo "=============================================="
echo ""

for bench_config in "${BENCHMARKS[@]}"; do
    IFS='|' read -r BENCHMARK SPLIT EXTRA <<< "$bench_config"

    for DIALECT in "${DIALECTS[@]}"; do
        COMPLETED=$((COMPLETED + 1))

        echo "=============================================="
        echo "[$COMPLETED/$TOTAL] $BENCHMARK x $DIALECT"
        echo "=============================================="

        # Check if this combination already exists (skip if so)
        EXISTING=$(ls data/benchmarks/${BENCHMARK}_${DIALECT}_*.json 2>/dev/null | head -1 || true)
        if [[ -n "$EXISTING" ]]; then
            echo "SKIP: Already exists: $EXISTING"
            echo ""
            continue
        fi

        python scripts/run_generation.py \
            --benchmark "$BENCHMARK" \
            --dialect "$DIALECT" \
            --backend ollama \
            --model "$MODEL" \
            --split "$SPLIT" \
            --end "$MAX_SAMPLES"

        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        AVG=$((ELAPSED / COMPLETED))
        REMAINING=$(( (TOTAL - COMPLETED) * AVG ))
        echo "Elapsed: $((ELAPSED / 60))m | ETA: $((REMAINING / 60))m"
        echo ""
    done
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=============================================="
echo "GENERATION COMPLETE"
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo "Results: data/benchmarks/"
echo ""
echo "Next step: run audits with:"
echo "  ./scripts/run_all_audits.sh"
echo "=============================================="
