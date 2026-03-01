#!/bin/bash
# Run fairness audits on all generated dialect pair datasets
# Automatically discovers all dataset files in data/benchmarks/
# Requires: venv activated, Ollama running with the target model
# Default model: gemma3:27b (override with MODEL env var)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source venv/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

MODEL="${MODEL:-gemma3:27b}"
echo "Using model: $MODEL"

# Auto-discover all dataset files (exclude archive/ and combined files)
mapfile -t DATASETS < <(find data/benchmarks -maxdepth 1 -name "*.json" -not -name "combined*" | sort)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: No datasets found in data/benchmarks/"
    echo "Run ./scripts/run_all_generations.sh first"
    exit 1
fi

TOTAL=${#DATASETS[@]}
COMPLETED=0
START_TIME=$(date +%s)

mkdir -p data/audits

echo "=============================================="
echo "Found $TOTAL datasets to audit"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    NAME=$(basename "$dataset" .json)

    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $NAME"
    echo "=============================================="

    # Check if audit already exists for this dataset
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
echo "AUDITS COMPLETE"
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo "Results: data/audits/"
echo ""
echo "Generate report:"
echo "  python -c \"from src.analysis import load_all_audits, generate_report; from pathlib import Path; audits = load_all_audits(Path('data/audits')); generate_report(audits, Path('data/reports'))\""
echo "=============================================="
