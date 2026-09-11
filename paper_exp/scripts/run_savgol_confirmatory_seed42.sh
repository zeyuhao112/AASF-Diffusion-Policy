#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Formal Savitzky-Golay confirmatory evaluation
#
# Frozen selected configuration:
#   method    = w5_p3
#   window    = 5
#   polyorder = 3
#
# Formal confirmatory seeds:
#   4300100-4300149
#
# IMPORTANT:
#   This script must be executed with:
#
#   SG_CONFIRMATORY_ARM=YES bash \
#       paper_exp/scripts/run_savgol_confirmatory_seed42.sh
#
#   Do NOT source this script.
# ============================================================

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "REFUSING TO RUN: do not source this launcher."
    echo "CONFIRMATORY_POLICY_ROLLOUT = NO"
    return 1
fi

if [ "${SG_CONFIRMATORY_ARM:-NO}" != "YES" ]; then
    echo "SG_CONFIRMATORY_ARM != YES"
    echo "FORMAL CONFIRMATORY RUN NOT STARTED"
    echo "CONFIRMATORY_POLICY_ROLLOUT = NO"
    exit 2
fi

cd /root/autodl-tmp/projects/diffusion_policy

SG_ROOT="paper_exp/savgol_baseline_seed42"

RUN_ROOT="$SG_ROOT/formal_confirmatory"

RESULT_ROOT="$RUN_ROOT/results"
TRACE_ROOT="$RUN_ROOT/traces"
LOG_ROOT="$RUN_ROOT/logs"
META_ROOT="$RUN_ROOT/metadata"

START_MARKER="$RUN_ROOT/FORMAL_CONFIRMATORY_STARTED.txt"
COMPLETE_MARKER="$RUN_ROOT/FORMAL_CONFIRMATORY_COMPLETE.txt"

ARTIFACT_MANIFEST="$RUN_ROOT/SG_CONFIRMATORY_ARTIFACT_SHA256.txt"

CHECKPOINT="data/checkpoints/pusht.ckpt"

RUNNER="diffusion_policy/env_runner/pusht_keypoints_runner.py"

PROTOCOL="$SG_ROOT/SG_PROTOCOL_FREEZE.txt"

SELECTED="$SG_ROOT/development_selection/SELECTED_SG_CONFIGURATION.txt"

SELECTION_MANIFEST="$SG_ROOT/development_selection/SG_SELECTION_SHA256.txt"

RECON="$SG_ROOT/freeze_reconciliation/SG_FREEZE_CHAIN_RECONCILIATION.txt"

CHAIN="$SG_ROOT/freeze_reconciliation/SG_FREEZE_CHAIN_SHA256.txt"

LAUNCHER_MANIFEST="$SG_ROOT/confirmatory_launcher/SG_CONFIRMATORY_LAUNCHER_SHA256.txt"

START_SEED=4300100
END_SEED=4300149
N_TEST=50

SELECTED_METHOD="w5_p3"
SELECTED_WINDOW=5
SELECTED_POLYORDER=3

sha_of() {
    sha256sum "$1" | awk '{print $1}'
}

echo "===== SG FORMAL CONFIRMATORY PREFLIGHT ====="

if [ -e "$RUN_ROOT" ]; then
    echo "FORMAL CONFIRMATORY RUN ROOT ALREADY EXISTS"
    echo "run_root=$RUN_ROOT"
    echo "REFUSING TO RERUN OR OVERWRITE"
    exit 1
fi

if ! sha256sum -c "$LAUNCHER_MANIFEST" >/dev/null 2>&1
then
    echo "CONFIRMATORY LAUNCHER MANIFEST VERIFY = FAIL"
    exit 1
fi

echo "CONFIRMATORY_LAUNCHER_MANIFEST_VERIFY = PASS"

if ! sha256sum -c "$CHAIN" >/dev/null 2>&1
then
    echo "FREEZE CHAIN VERIFY = FAIL"
    exit 1
fi

echo "SG_FREEZE_CHAIN_MANIFEST_VERIFY = PASS"

if ! sha256sum -c "$SELECTION_MANIFEST" >/dev/null 2>&1
then
    echo "SELECTION MANIFEST VERIFY = FAIL"
    exit 1
fi

echo "SG_SELECTION_MANIFEST_VERIFY = PASS"

grep -Fqx \
    "method=w5_p3" \
    "$SELECTED"

grep -Fqx \
    "window=5" \
    "$SELECTED"

grep -Fqx \
    "polyorder=3" \
    "$SELECTED"

grep -Fqx \
    "development_selection_frozen=YES" \
    "$SELECTED"

grep -Fqx \
    "confirmatory_data_accessed=NO" \
    "$SELECTED"

grep -Fqx \
    "post_unblind_retuning_permitted=NO" \
    "$SELECTED"

echo "FROZEN_SELECTED_CONFIG = PASS"

python -m py_compile "$RUNNER"

echo "RUNNER_PYCOMPILE = PASS"

if ! sha256sum -c \
    paper_submission_final_frozen_20260816.tar.gz.sha256.txt \
    >/dev/null 2>&1
then
    echo "FROZEN SUBMISSION ARCHIVE VERIFY = FAIL"
    exit 1
fi

echo "FROZEN_SUBMISSION_ARCHIVE = PASS"

# ============================================================
# Formal unblinding starts only after every preflight passes.
# ============================================================

mkdir -p \
    "$RUN_ROOT" \
    "$TRACE_ROOT" \
    "$LOG_ROOT" \
    "$META_ROOT"

printf '%s\n' \
    "study=savgol_formal_confirmatory_seed42" \
    "status=STARTED" \
    "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "formal_confirmatory_unblinded=YES" \
    "start_seed=$START_SEED" \
    "end_seed=$END_SEED" \
    "n_test=$N_TEST" \
    "selected_method=$SELECTED_METHOD" \
    "selected_window=$SELECTED_WINDOW" \
    "selected_polyorder=$SELECTED_POLYORDER" \
    "alternative_sg_configurations_permitted=NO" \
    "post_unblind_retuning_permitted=NO" \
    > "$START_MARKER"

echo
echo "===== FORMAL CONFIRMATORY UNBLINDING STARTED ====="
echo "seeds=$START_SEED-$END_SEED"
echo "selected_sg=$SELECTED_METHOD"
echo "alternative_sg_configs=FORBIDDEN"

# ============================================================
# Shared environment
# ============================================================

export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline

export DP_N_ENVS=1
export DP_N_TEST="$N_TEST"
export DP_N_TEST_VIS=0
export DP_N_TRAIN=0
export DP_N_TRAIN_VIS=0
export DP_TEST_START_SEED="$START_SEED"

export DP_PAIRED_POLICY_SEED=1
export DP_POLICY_SEED_SALT=20260710

# Strict isolation.
export DP_BSOP_ENABLE=0

export DP_TE_ENABLE=0
export DP_TE_BETA=0.0
export DP_TE_MAX_HISTORY=1

export DP_ACTION_SMOOTH_ALPHA=1.0
export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
export DP_ACTION_SMOOTH_ALPHA_MAX=0.8
export DP_ACTION_SMOOTH_THRESHOLD=10.0
export DP_ACTION_SMOOTH_GAIN=0.25

unset DP_ACTION_SMA_WINDOW || true

# Full execution-accounting traces.
export DP_TRACE_ENABLE=1
export DP_TRACE_WARMUP_QUERIES=3
export DP_OVERLAP_DIAG_ENABLE=1

# ============================================================
# One formal method
# ============================================================

run_method() {
    local method="$1"
    local mode="$2"
    local window="$3"
    local polyorder="$4"

    local result_dir="$RESULT_ROOT/$method"
    local trace_dir="$TRACE_ROOT/$method"
    local log="$LOG_ROOT/$method.log"
    local meta="$META_ROOT/$method.txt"

    echo
    echo "============================================================"
    echo "FORMAL METHOD: $method"
    echo "mode=$mode"
    echo "window=$window"
    echo "polyorder=$polyorder"
    echo "============================================================"

    if [ -e "$result_dir" ] \
        || [ -e "$trace_dir" ] \
        || [ -e "$log" ] \
        || [ -e "$meta" ]
    then
        echo "METHOD ARTIFACT ALREADY EXISTS"
        echo "REFUSING TO OVERWRITE"
        exit 1
    fi

    mkdir -p "$trace_dir"

    export DP_ACTION_SMOOTH_MODE="$mode"
    export DP_ACTION_SG_WINDOW="$window"
    export DP_ACTION_SG_POLYORDER="$polyorder"

    export DP_TRACE_DIR="$PWD/$trace_dir"
    export DP_TRACE_METHOD="sg_confirmatory_seed42_${method}"

    {
        echo "study=savgol_formal_confirmatory_seed42"
        echo "method=$method"
        echo "status=started"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        echo "checkpoint=$CHECKPOINT"
        echo "checkpoint_sha256=$(sha_of "$CHECKPOINT")"
        echo "runner_sha256=$(sha_of "$RUNNER")"
        echo "protocol_sha256=$(sha_of "$PROTOCOL")"
        echo "selected_config_sha256=$(sha_of "$SELECTED")"
        echo "selection_manifest_sha256=$(sha_of "$SELECTION_MANIFEST")"
        echo "freeze_chain_sha256=$(sha_of "$CHAIN")"
        echo
        echo "start_seed=$START_SEED"
        echo "end_seed=$END_SEED"
        echo "n_test=$N_TEST"
        echo "n_envs=1"
        echo
        echo "DP_PAIRED_POLICY_SEED=$DP_PAIRED_POLICY_SEED"
        echo "DP_POLICY_SEED_SALT=$DP_POLICY_SEED_SALT"
        echo
        echo "DP_BSOP_ENABLE=$DP_BSOP_ENABLE"
        echo "DP_TE_ENABLE=$DP_TE_ENABLE"
        echo
        echo "DP_ACTION_SMOOTH_MODE=$DP_ACTION_SMOOTH_MODE"
        echo "DP_ACTION_SMOOTH_ALPHA=$DP_ACTION_SMOOTH_ALPHA"
        echo "DP_ACTION_SG_WINDOW=$DP_ACTION_SG_WINDOW"
        echo "DP_ACTION_SG_POLYORDER=$DP_ACTION_SG_POLYORDER"
        echo
    } > "$meta"

    set +e

    python eval.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$result_dir" \
        --device cuda:0 \
        2>&1 | tee "$log"

    status=${PIPESTATUS[0]}

    set -e

    {
        echo
        echo "python_exit_status=$status"
    } >> "$meta"

    if [ "$status" -ne 0 ]; then
        {
            echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "status=FAILED"
        } >> "$meta"

        echo "METHOD PROCESS FAILED: $method"
        exit "$status"
    fi

    normal_count=$(
        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name 'test_seed_*.npz' \
            ! -name '*_overlap_diag.npz' \
            | wc -l
    )

    diag_count=$(
        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name '*_overlap_diag.npz' \
            | wc -l
    )

    echo "normal_trace_count=$normal_count" >> "$meta"
    echo "diagnostic_trace_count=$diag_count" >> "$meta"

    if [ "$normal_count" -ne "$N_TEST" ]; then
        echo "NORMAL TRACE COUNT FAILURE"
        exit 1
    fi

    if [ "$diag_count" -ne "$N_TEST" ]; then
        echo "DIAGNOSTIC TRACE COUNT FAILURE"
        exit 1
    fi

    if grep -nEi \
        'Traceback|RuntimeError|ValueError|CUDA out of memory|Killed|alignment.*failed|cursor.*failed' \
        "$log"
    then
        echo "ERROR MARKER FOUND: $method"
        exit 1
    fi

    if grep -q '\[temporal-ensemble\]' "$log"; then
        echo "TEMPORAL ENSEMBLE ISOLATION FAILURE"
        exit 1
    fi

    if ! grep -Fq \
        "test_start_seed=4300100, n_test=50, n_train=0" \
        "$log"
    then
        echo "FORMAL SEED OVERRIDE MARKER FAILURE"
        exit 1
    fi

    {
        echo
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "status=completed"
        echo
        echo "[Trace SHA256]"

        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name '*.npz' \
            -print0 \
            | sort -z \
            | xargs -0 sha256sum
    } >> "$meta"

    echo "$method = PASS"
}

# ============================================================
# Exactly two formal arms:
#
#   1. fresh unsmoothed Original reference
#   2. the ONE mechanically selected SG configuration
#
# No other SG configuration may be evaluated.
# ============================================================

run_method \
    "original" \
    "fixed" \
    "5" \
    "3"

run_method \
    "w5_p3" \
    "sg" \
    "5" \
    "3"

# ============================================================
# Formal artifact accounting
# ============================================================

TOTAL_NORMAL=$(
    find "$TRACE_ROOT" \
        -type f \
        -name 'test_seed_*.npz' \
        ! -name '*_overlap_diag.npz' \
        | wc -l
)

TOTAL_DIAG=$(
    find "$TRACE_ROOT" \
        -type f \
        -name '*_overlap_diag.npz' \
        | wc -l
)

if [ "$TOTAL_NORMAL" -ne 100 ]; then
    echo "FORMAL NORMAL TRACE ACCOUNTING FAILURE"
    exit 1
fi

if [ "$TOTAL_DIAG" -ne 100 ]; then
    echo "FORMAL DIAGNOSTIC TRACE ACCOUNTING FAILURE"
    exit 1
fi

{
    sha256sum \
        "$CHECKPOINT" \
        "$RUNNER" \
        "$PROTOCOL" \
        "$SELECTED" \
        "$SELECTION_MANIFEST" \
        "$RECON" \
        "$CHAIN" \
        "$LAUNCHER_MANIFEST" \
        "$START_MARKER"

    find \
        "$RESULT_ROOT" \
        "$TRACE_ROOT" \
        "$LOG_ROOT" \
        "$META_ROOT" \
        -type f \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum

} > "$ARTIFACT_MANIFEST"

printf '%s\n' \
    "study=savgol_formal_confirmatory_seed42" \
    "status=COMPLETE" \
    "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "formal_confirmatory_unblinded=YES" \
    "start_seed=$START_SEED" \
    "end_seed=$END_SEED" \
    "n_original=50" \
    "n_selected_sg=50" \
    "selected_method=w5_p3" \
    "selected_window=5" \
    "selected_polyorder=3" \
    "alternative_sg_configurations_evaluated=NO" \
    "post_unblind_retuning_permitted=NO" \
    "artifact_manifest=$ARTIFACT_MANIFEST" \
    "artifact_manifest_sha256=$(sha_of "$ARTIFACT_MANIFEST")" \
    > "$COMPLETE_MARKER"

echo
echo "===== SG FORMAL CONFIRMATORY COMPLETE ====="
echo "original_episodes=50"
echo "selected_sg_episodes=50"
echo "selected_method=w5_p3"
echo "selected_window=5"
echo "selected_polyorder=3"
echo "alternative_sg_configs_evaluated=NO"
echo "post_unblind_retuning_permitted=NO"
echo "SG_FORMAL_CONFIRMATORY_RUN = PASS"
