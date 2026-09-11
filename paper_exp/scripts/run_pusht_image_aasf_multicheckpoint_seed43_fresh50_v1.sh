#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/diffusion_policy

RUNNER="diffusion_policy/env_runner/pusht_image_runner.py"
CKPT="data/checkpoints/pusht_image_official/epoch=1450-test_mean_score=0.897.ckpt"

EXPECTED_RUNNER_SHA="8f0a009d55e80ff586c626e1f6f5667227eb66952ae4e6542fbcccf371bbe4da"
EXPECTED_CHECKPOINT_SHA="7d5824bfaa95358e8f31921b8b5d46e7603dcf320053111785c440161df804dd"

START_SEED=4700100
END_SEED=4700149
N_TEST=50
CONDITION="Push-T image seed43 / epoch1450"

RUN_NAME="pusht_image_aasf_multicheckpoint_seed43_fresh50"

RESULT_ROOT="paper_exp/results/$RUN_NAME"
TRACE_ROOT="paper_artifacts/traces/$RUN_NAME"
LOG_ROOT="paper_exp/logs/$RUN_NAME"
META_ROOT="paper_exp/metadata/$RUN_NAME"

echo "============================================================"
echo "AASF IMAGE MULTI-CHECKPOINT REPLICATION"
echo "condition=$CONDITION"
echo "run_name=$RUN_NAME"
echo "============================================================"

ACTUAL_RUNNER_SHA="$(sha256sum "$RUNNER" | awk '{print $1}')"
ACTUAL_CHECKPOINT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"

echo "runner expected = $EXPECTED_RUNNER_SHA"
echo "runner actual   = $ACTUAL_RUNNER_SHA"
echo "ckpt expected   = $EXPECTED_CHECKPOINT_SHA"
echo "ckpt actual     = $ACTUAL_CHECKPOINT_SHA"

if [ "$ACTUAL_RUNNER_SHA" != "$EXPECTED_RUNNER_SHA" ]; then
    echo "AASF_IMAGE_RUNNER_IDENTITY = FAIL"
    exit 1
fi

if [ "$ACTUAL_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ]; then
    echo "AASF_IMAGE_CHECKPOINT_IDENTITY = FAIL"
    exit 1
fi

python -m py_compile "$RUNNER"

if [ "${AASF_IMAGE_REPLICATION_ARM:-NO}" != "YES" ]; then
    echo "AASF_IMAGE_REPLICATION_ARM != YES"
    echo "No output root will be created."
    exit 2
fi

if [ -e "$RESULT_ROOT" ] || \
   [ -e "$TRACE_ROOT" ] || \
   [ -e "$META_ROOT" ]; then

    echo "ERROR: output already exists."
    echo "REFUSING_TO_OVERWRITE = YES"
    echo "  $RESULT_ROOT"
    echo "  $TRACE_ROOT"
    echo "  $META_ROOT"
    exit 1
fi

mkdir -p \
    "$RESULT_ROOT" \
    "$TRACE_ROOT" \
    "$LOG_ROOT" \
    "$META_ROOT"

SEED_LIST="$(seq -s, "$START_SEED" "$END_SEED")"

cat > "$META_ROOT/master_run_metadata.txt" <<META
status=started
condition=$CONDITION
run_name=$RUN_NAME

start_seed=$START_SEED
end_seed=$END_SEED
n_test=$N_TEST

runner_sha256=$EXPECTED_RUNNER_SHA
checkpoint=$CKPT
checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA

paired_policy_seed=1
policy_seed_salt=20260710
DP_N_ENVS=1

AASF frozen parameters:
DP_ACTION_SMOOTH_MODE=adaptive
DP_ACTION_SMOOTH_ALPHA=0.80
DP_ACTION_SMOOTH_ALPHA_MIN=0.45
DP_ACTION_SMOOTH_ALPHA_MAX=0.80
DP_ACTION_SMOOTH_THRESHOLD=10.0
DP_ACTION_SMOOTH_GAIN=0.25

BSOP_enabled=NO
image_specific_retuning=NO
checkpoint_specific_retuning=NO
META

clear_dp_vars () {
    while IFS='=' read -r name _; do
        case "$name" in
            DP_*) unset "$name" ;;
        esac
    done < <(env)
}

run_method () {
    METHOD="$1"

    RESULT="$RESULT_ROOT/$METHOD"
    TRACE="$TRACE_ROOT/$METHOD"
    LOG="$LOG_ROOT/${METHOD}.txt"
    METHOD_META="$META_ROOT/${METHOD}_run_metadata.txt"

    mkdir -p "$TRACE"

    clear_dp_vars

    export DP_TEST_START_SEED="$START_SEED"
    export DP_N_TEST="$N_TEST"
    export DP_N_TEST_VIS=0
    export DP_N_TRAIN=0
    export DP_N_TRAIN_VIS=0

    export DP_N_ENVS=1
    export DP_PAIRED_POLICY_SEED=1
    export DP_POLICY_SEED_SALT=20260710

    export DP_IMAGE_TRACE=1
    export DP_IMAGE_TRACE_DIR="$TRACE"
    export DP_IMAGE_TRACE_METHOD="$METHOD"
    export DP_IMAGE_TRACE_SEEDS="$SEED_LIST"

    case "$METHOD" in
        original)
            unset DP_BSOP_ENABLE
            unset DP_ACTION_SMOOTH_MODE
            unset DP_ACTION_SMOOTH_ALPHA
            unset DP_ACTION_SMOOTH_ALPHA_MIN
            unset DP_ACTION_SMOOTH_ALPHA_MAX
            unset DP_ACTION_SMOOTH_THRESHOLD
            unset DP_ACTION_SMOOTH_GAIN
            ;;
        aasf)
            unset DP_BSOP_ENABLE
            export DP_ACTION_SMOOTH_MODE=adaptive
            export DP_ACTION_SMOOTH_ALPHA=0.80
            export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
            export DP_ACTION_SMOOTH_ALPHA_MAX=0.80
            export DP_ACTION_SMOOTH_THRESHOLD=10.0
            export DP_ACTION_SMOOTH_GAIN=0.25
            ;;
        *)
            echo "ERROR: unknown method $METHOD"
            exit 1
            ;;
    esac

    cat > "$METHOD_META" <<META
method=$METHOD
status=started
condition=$CONDITION
start_seed=$START_SEED
end_seed=$END_SEED
n_test=$N_TEST
runner_sha256=$EXPECTED_RUNNER_SHA
checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA
DP_N_ENVS=1
DP_PAIRED_POLICY_SEED=1
DP_POLICY_SEED_SALT=20260710
BSOP_enabled=NO
META

    if [ "$METHOD" = "aasf" ]; then
        cat >> "$METHOD_META" <<META
DP_ACTION_SMOOTH_MODE=adaptive
DP_ACTION_SMOOTH_ALPHA=0.80
DP_ACTION_SMOOTH_ALPHA_MIN=0.45
DP_ACTION_SMOOTH_ALPHA_MAX=0.80
DP_ACTION_SMOOTH_THRESHOLD=10.0
DP_ACTION_SMOOTH_GAIN=0.25
image_specific_retuning=NO
checkpoint_specific_retuning=NO
META
    fi

    echo
    echo "============================================================"
    echo "METHOD: $METHOD"
    echo "SEEDS: $START_SEED-$END_SEED"
    echo "============================================================"

    env | grep '^DP_' | sort

    set +e
    python eval.py \
        --checkpoint "$CKPT" \
        --output_dir "$RESULT" \
        --device cuda:0 \
        2>&1 | tee "$LOG"

    RC=${PIPESTATUS[0]}
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "status=failed" >> "$METHOD_META"
        echo "ERROR: $METHOD failed with rc=$RC"
        exit "$RC"
    fi

    if grep -Eqi \
        'traceback|runtimeerror|assertionerror|cuda out of memory|error:' \
        "$LOG"
    then
        echo "status=failed_error_scan" >> "$METHOD_META"
        echo "ERROR: failure term found in $LOG"
        exit 1
    fi

    TRACE_COUNT="$(
        find "$TRACE" \
            -maxdepth 1 \
            -type f \
            -name '*.npz' \
            | wc -l
    )"

    echo "trace_count=$TRACE_COUNT" >> "$METHOD_META"

    if [ "$TRACE_COUNT" -ne "$N_TEST" ]; then
        echo "status=failed_trace_count" >> "$METHOD_META"
        echo "ERROR: $METHOD has $TRACE_COUNT traces; expected $N_TEST."
        exit 1
    fi

    METHOD="$METHOD" \
    TRACE="$TRACE" \
    START_SEED="$START_SEED" \
    END_SEED="$END_SEED" \
    python - <<'PY'
import os
from pathlib import Path
import numpy as np

method = os.environ["METHOD"]
root = Path(os.environ["TRACE"])
start = int(os.environ["START_SEED"])
end = int(os.environ["END_SEED"])

overall = True

for seed in range(start, end + 1):
    p = root / f"seed_{seed}_{method}.npz"

    if not p.exists():
        print("MISSING:", p)
        overall = False
        continue

    with np.load(p, allow_pickle=True) as d:
        chunks = np.asarray(d["executed_action"])
        executed = np.asarray(
            d["executed_steps_per_query"],
            dtype=np.int64,
        )
        actual = np.asarray(
            d["actual_executed_action"]
        )

        parts = []

        for q, n in enumerate(executed):
            if n > 0:
                parts.append(chunks[q, :n])

        rebuilt = np.concatenate(parts, axis=0)

        ok = (
            rebuilt.shape == actual.shape
            and np.array_equal(rebuilt, actual)
        )

        if not ok:
            print("EXECUTION ACCOUNTING FAILED:", seed)
            overall = False

if not overall:
    raise SystemExit(1)

print("Execution accounting passed for", end - start + 1, "seeds.")
PY

    echo "status=completed" >> "$METHOD_META"

    echo
    echo "$METHOD completed successfully."
}

run_method original
run_method aasf

echo "status=completed" >> "$META_ROOT/master_run_metadata.txt"

echo
echo "============================================================"
echo "AASF IMAGE REPLICATION RUN COMPLETED"
echo "condition=$CONDITION"
echo "run_name=$RUN_NAME"
echo "============================================================"
