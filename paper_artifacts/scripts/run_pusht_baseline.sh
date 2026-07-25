#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pusht_baseline.sh METHOD START_SEED N_TEST OUTPUT_DIR
#
# METHOD:
#   original | sma3 | sma5 | fixed | aasf
#
# Example:
#   bash run_pusht_baseline.sh sma3 4300050 50 \
#     paper_exp/results/sma_w3_seed4300050

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 METHOD START_SEED N_TEST OUTPUT_DIR"
    exit 1
fi

METHOD="$1"
START_SEED="$2"
N_TEST="$3"
OUTPUT_DIR="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT="data/checkpoints/pusht.ckpt"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTPUT_DIR"
    echo "Choose a new directory or remove it manually after checking its contents."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p paper_exp/logs

# Headless and offline evaluation.
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline

# Evaluation episodes.
export DP_TEST_START_SEED="$START_SEED"
export DP_N_TEST="$N_TEST"
export DP_N_TEST_VIS=0
export DP_N_TRAIN=0
export DP_N_TRAIN_VIS=0

# Clear every smoothing-related variable first.
unset DP_ACTION_SMOOTH_MODE || true
unset DP_ACTION_SMOOTH_ALPHA || true
unset DP_ACTION_SMA_WINDOW || true
unset DP_ACTION_SMOOTH_ALPHA_MIN || true
unset DP_ACTION_SMOOTH_ALPHA_MAX || true
unset DP_ACTION_SMOOTH_THRESHOLD || true
unset DP_ACTION_SMOOTH_GAIN || true

case "$METHOD" in
    original)
        # Guard condition is false, so no smoothing is applied.
        export DP_ACTION_SMOOTH_MODE=fixed
        export DP_ACTION_SMOOTH_ALPHA=1.0
        ;;

    sma3)
        export DP_ACTION_SMOOTH_MODE=sma
        export DP_ACTION_SMOOTH_ALPHA=1.0
        export DP_ACTION_SMA_WINDOW=3
        ;;

    sma5)
        export DP_ACTION_SMOOTH_MODE=sma
        export DP_ACTION_SMOOTH_ALPHA=1.0
        export DP_ACTION_SMA_WINDOW=5
        ;;

    fixed)
        export DP_ACTION_SMOOTH_MODE=fixed
        export DP_ACTION_SMOOTH_ALPHA=0.7
        ;;

    aasf)
        export DP_ACTION_SMOOTH_MODE=adaptive
        export DP_ACTION_SMOOTH_ALPHA=0.8
        export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
        export DP_ACTION_SMOOTH_ALPHA_MAX=0.8
        export DP_ACTION_SMOOTH_THRESHOLD=10.0
        export DP_ACTION_SMOOTH_GAIN=0.25
        ;;

    *)
        echo "ERROR: unknown method '$METHOD'"
        echo "Valid methods: original, sma3, sma5, fixed, aasf"
        exit 1
        ;;
esac

SAFE_OUTPUT_NAME="$(echo "$OUTPUT_DIR" | tr '/ ' '__')"
LOG_FILE="paper_exp/logs/${SAFE_OUTPUT_NAME}.log"
STAGING_METADATA_FILE="paper_exp/logs/${SAFE_OUTPUT_NAME}.run_metadata.txt"
FINAL_METADATA_FILE="${OUTPUT_DIR}/run_metadata.txt"

{
    echo "method=$METHOD"
    echo "start_seed=$START_SEED"
    echo "n_test=$N_TEST"
    echo "output_dir=$OUTPUT_DIR"
    echo "checkpoint=$CHECKPOINT"
    echo "device=$DEVICE"
    echo "python_bin=$PYTHON_BIN"
    echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "[Git]"
    git rev-parse HEAD 2>/dev/null || true
    git status --porcelain 2>/dev/null || true
    echo
    echo "[Checkpoint SHA256]"
    sha256sum "$CHECKPOINT"
    echo
    echo "[Locked baseline definition SHA256]"
    sha256sum paper_artifacts/code/baseline_definitions_locked.md
    echo
    echo "[Environment variables]"
    env | grep -E \
        '^(DP_[A-Za-z0-9_]*|WANDB_MODE|SDL_VIDEODRIVER)=' \
        | sort
    echo
    echo "[Python]"
    "$PYTHON_BIN" --version
    echo
    echo "[GPU]"
    nvidia-smi 2>/dev/null || true
} | tee "$STAGING_METADATA_FILE"

echo
echo "Starting evaluation..."
echo "Log: $LOG_FILE"
echo

"$PYTHON_BIN" eval.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_FILE"

{
    echo
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} | tee -a "$STAGING_METADATA_FILE"

if [[ ! -f "${OUTPUT_DIR}/eval_log.json" ]]; then
    echo "ERROR: evaluation finished without eval_log.json"
    echo "Pre-run metadata remains at:"
    echo "  $STAGING_METADATA_FILE"
    exit 1
fi

cp "$STAGING_METADATA_FILE" "$FINAL_METADATA_FILE"

sha256sum \
    "${OUTPUT_DIR}/eval_log.json" \
    "$FINAL_METADATA_FILE" \
    | tee "${OUTPUT_DIR}/result_checksums.sha256"

echo
echo "Evaluation completed successfully:"
echo "  ${OUTPUT_DIR}/eval_log.json"
