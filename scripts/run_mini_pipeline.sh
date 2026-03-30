#!/bin/bash
# Full pipeline for Mac Mini: generation top-up, top-up audits, full 5th model audit.
# Swaps models between stages to save RAM.
# Run: ./scripts/run_mini_pipeline.sh 2>&1 | tee mini_pipeline.log
cd ~/Projects/Capstone
source venv/bin/activate
export PYTHONPATH=~/Projects/Capstone

START_TIME=$(date +%s)

log() {
    echo ""
    echo "=============================================="
    echo "[$(date '+%H:%M:%S')] $1"
    echo "=============================================="
}

run_audit() {
    local pairs=$1 model=$2
    local model_safe=$(echo "$model" | tr ':/' '__')
    mkdir -p data/audits/$model_safe
    python3 scripts/run_audit.py --pairs "$pairs" --backend ollama --model "$model"
}

# Top-up pair files (300-600 range, the ones with actual data)
TOPUP_FILES=(
    data/benchmarks/boolq_aave_300-600_20260329_163531.json
    data/benchmarks/donotanswer_aave_300-600_20260329_172911.json
    data/benchmarks/donotanswer_hiberno_english_300-600_20260329_175921.json
    data/benchmarks/mmlu_aave_300-600_20260329_161031.json
    data/benchmarks/realtoxicityprompts_aave_300-600_20260329_165537.json
    data/benchmarks/truthfulqa_aave_300-600_20260329_175335.json
    data/benchmarks/truthfulqa_hiberno_english_300-600_20260329_180533.json
)

# All 27 benchmark files (for full 5th model audit)
ALL_FILES=(
    data/benchmarks/arc_aave_0-300_20260301_235243.json
    data/benchmarks/arc_hiberno_english_0-300_20260301_195955.json
    data/benchmarks/arc_indian_english_0-300_20260303_102245.json
    data/benchmarks/boolq_aave_combined.json
    data/benchmarks/boolq_hiberno_english_0-300_20260301_205235.json
    data/benchmarks/boolq_indian_english_0-300_20260304_010831.json
    data/benchmarks/donotanswer_aave_combined.json
    data/benchmarks/donotanswer_hiberno_english_combined.json
    data/benchmarks/donotanswer_indian_english_0-300_20260304_023449.json
    data/benchmarks/gsm8k_aave_0-300_20260301_221903.json
    data/benchmarks/gsm8k_hiberno_english_0-300_20260301_193657.json
    data/benchmarks/gsm8k_indian_english_0-300_20260303_091722.json
    data/benchmarks/hellaswag_aave_0-300_20260302_002727.json
    data/benchmarks/hellaswag_hiberno_english_0-300_20260301_201241.json
    data/benchmarks/hellaswag_indian_english_0-300_20260303_110000.json
    data/benchmarks/mmlu_aave_combined.json
    data/benchmarks/mmlu_hiberno_english_0-300_20260301_194748.json
    data/benchmarks/mmlu_indian_english_0-300_20260303_094652.json
    data/benchmarks/realtoxicityprompts_aave_combined.json
    data/benchmarks/realtoxicityprompts_hiberno_english_0-300_20260301_210829.json
    data/benchmarks/realtoxicityprompts_indian_english_0-300_20260304_020748.json
    data/benchmarks/toxigen_aave_0-300_20260302_160758.json
    data/benchmarks/toxigen_hiberno_english_0-300_20260301_212923.json
    data/benchmarks/toxigen_indian_english_0-300_20260304_032719.json
    data/benchmarks/truthfulqa_aave_combined.json
    data/benchmarks/truthfulqa_hiberno_english_combined.json
    data/benchmarks/truthfulqa_indian_english_0-300_20260304_012840.json
)

AUDIT_MODELS=(gemma2:9b llama3.1:8b mistral:7b qwen2.5:14b)

# ============================================================
# PHASE 1: Generate ARC AAVE top-up (qwen2.5:14b already loaded)
# ============================================================
log "PHASE 1: ARC AAVE top-up generation"
python3 scripts/run_generation.py \
    --benchmark arc --dialect aave \
    --backend ollama --model qwen2.5:14b \
    --split test --start 300 --end 600

# Find the file that was just created
ARC_TOPUP=$(ls -t data/benchmarks/arc_aave_300-600_*.json 2>/dev/null | head -1)
if [[ -n "$ARC_TOPUP" ]]; then
    echo "ARC top-up saved to: $ARC_TOPUP"
    TOPUP_FILES+=("$ARC_TOPUP")
else
    echo "WARNING: ARC top-up file not found, continuing without it"
fi

# ============================================================
# PHASE 2: Top-up audits (new pairs only, cycle through models)
# ============================================================
for model in "${AUDIT_MODELS[@]}"; do
    log "PHASE 2: Pulling $model for top-up audits"
    ollama pull "$model"

    for pairs_file in "${TOPUP_FILES[@]}"; do
        name=$(basename "$pairs_file" .json)
        log "Auditing $name with $model"
        run_audit "$pairs_file" "$model" || echo "ERROR: $name x $model failed, continuing..."
    done

    log "Removing $model"
    ollama rm "$model"
done

# ============================================================
# PHASE 3: Full Qwen 2.5 7B audit (new 5th model)
# ============================================================
log "PHASE 3: Pulling qwen2.5:7b for full audit"
ollama pull qwen2.5:7b

for pairs_file in "${ALL_FILES[@]}"; do
    name=$(basename "$pairs_file" .json)
    log "Auditing $name with qwen2.5:7b"
    run_audit "$pairs_file" "qwen2.5:7b" || echo "ERROR: $name x qwen2.5:7b failed, continuing..."
done

# ============================================================
# DONE
# ============================================================
END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
log "ALL DONE in $((TOTAL / 3600))h $((TOTAL % 3600 / 60))m"
