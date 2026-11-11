#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/diffusion_policy

RUNNER="diffusion_policy/env_runner/pusht_image_runner.py"
CKPT="data/checkpoints/pusht_image_official/epoch=1450-test_mean_score=0.897.ckpt"

EXPECTED_RUNNER_SHA="0bcf6c5c3158f3fc0697a4c75c008a656555053be0d9e961fd5c9bb565547f20"
EXPECTED_CHECKPOINT_SHA="7d5824bfaa95358e8f31921b8b5d46e7603dcf320053111785c440161df804dd"

START_SEED=4500300
END_SEED=4500349
N_TEST=50

RUN_NAME="pusht_image_bsop_multicheckpoint_seed43_fresh50"

RESULT_ROOT="paper_exp/results/$RUN_NAME"
TRACE_ROOT="paper_artifacts/traces/$RUN_NAME"
LOG_ROOT="paper_exp/logs/$RUN_NAME"
META_ROOT="paper_exp/metadata/$RUN_NAME"

echo "===== FROZEN IDENTITY CHECK ====="

ACTUAL_RUNNER_SHA="$(
    sha256sum "$RUNNER" | awk '{print $1}'
)"

ACTUAL_CHECKPOINT_SHA="$(
    sha256sum "$CKPT" | awk '{print $1}'
)"

echo "runner expected = $EXPECTED_RUNNER_SHA"
echo "runner actual   = $ACTUAL_RUNNER_SHA"
echo
echo "ckpt expected   = $EXPECTED_CHECKPOINT_SHA"
echo "ckpt actual     = $ACTUAL_CHECKPOINT_SHA"

if [ "$ACTUAL_RUNNER_SHA" != "$EXPECTED_RUNNER_SHA" ]; then
    echo "ERROR: runner SHA changed."
    exit 1
fi

if [ "$ACTUAL_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ]; then
    echo "ERROR: checkpoint SHA changed."
    exit 1
fi

python -m py_compile "$RUNNER"

if [ -e "$RESULT_ROOT" ] || [ -e "$TRACE_ROOT" ] || [ -e "$META_ROOT" ]; then
    echo
    echo "ERROR: confirmatory output already exists."
    echo "Refusing to overwrite:"
    echo "  $RESULT_ROOT"
    echo "  $TRACE_ROOT"
    echo "  $META_ROOT"
    exit 1
fi

mkdir -p     "$RESULT_ROOT"     "$TRACE_ROOT"     "$LOG_ROOT"     "$META_ROOT"

SEED_LIST="$(
    seq -s, "$START_SEED" "$END_SEED"
)"

cat > "$META_ROOT/master_run_metadata.txt" <<EOF
status=started
run_name=$RUN_NAME
start_seed=$START_SEED
end_seed=$END_SEED
n_test=$N_TEST
runner_sha256=$EXPECTED_RUNNER_SHA
checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA
paired_policy_seed=1
policy_seed_salt=20260710

BSOP frozen Push-T parameters:
DP_BSOP_ETA=1.0
DP_BSOP_LAMBDA_MAX=0.5
DP_BSOP_RHO=5.0
DP_BSOP_KAPPA=0.5

This is a modality-transfer confirmatory evaluation.
No Image-specific BSOP retuning is permitted.
EOF

run_method () {
    METHOD="$1"

    RESULT="$RESULT_ROOT/$METHOD"
    TRACE="$TRACE_ROOT/$METHOD"
    LOG="$LOG_ROOT/${METHOD}.txt"
    METHOD_META="$META_ROOT/${METHOD}_run_metadata.txt"

    mkdir -p "$TRACE"

    # Clear all stale experiment controls first.
    while IFS='=' read -r name _; do
        case "$name" in
            DP_*) unset "$name" ;;
        esac
    done < <(env)

    export DP_TEST_START_SEED="$START_SEED"
    export DP_N_TEST="$N_TEST"
    export DP_N_TEST_VIS=0
    export DP_N_TRAIN=0
    export DP_N_TRAIN_VIS=0
    export DP_N_ENVS=1

    # Common random numbers.
    export DP_PAIRED_POLICY_SEED=1
    export DP_POLICY_SEED_SALT=20260710

    # Full forensic / actual-execution trace.
    export DP_IMAGE_TRACE=1
    export DP_IMAGE_TRACE_DIR="$TRACE"
    export DP_IMAGE_TRACE_METHOD="$METHOD"
    export DP_IMAGE_TRACE_SEEDS="$SEED_LIST"

    case "$METHOD" in
        original)
            unset DP_BSOP_ENABLE
            ;;

        bsop)
            export DP_BSOP_ENABLE=1
            export DP_BSOP_ETA=1.0
            export DP_BSOP_LAMBDA_MAX=0.5
            export DP_BSOP_RHO=5.0
            export DP_BSOP_KAPPA=0.5
            ;;

        *)
            echo "Unknown method: $METHOD"
            exit 1
            ;;
    esac

    cat > "$METHOD_META" <<EOF
method=$METHOD
status=started
start_seed=$START_SEED
end_seed=$END_SEED
n_test=$N_TEST
runner_sha256=$EXPECTED_RUNNER_SHA
checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA
DP_PAIRED_POLICY_SEED=1
DP_POLICY_SEED_SALT=20260710
DP_BSOP_ETA=1.0
DP_BSOP_LAMBDA_MAX=0.5
DP_BSOP_RHO=5.0
DP_BSOP_KAPPA=0.5
EOF

    echo
    echo "============================================================"
    echo "PUSHT IMAGE CONFIRMATORY METHOD: $METHOD"
    echo "Seeds: $START_SEED-$END_SEED"
    echo "============================================================"

    env | grep '^DP_' | sort

    set +e
    python eval.py         --checkpoint "$CKPT"         --output_dir "$RESULT"         --device cuda:0         2>&1 | tee "$LOG"

    RC=${PIPESTATUS[0]}
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "status=failed" >> "$METHOD_META"
        echo "$METHOD failed with rc=$RC"
        exit "$RC"
    fi

    if grep -Eqi         'traceback|runtimeerror|assertionerror|cuda out of memory|error:'         "$LOG"
    then
        echo "ERROR: failure term found in $LOG"
        echo "status=failed_error_scan" >> "$METHOD_META"
        exit 1
    fi

    TRACE_COUNT="$(
        find "$TRACE"             -maxdepth 1             -type f             -name '*.npz'             | wc -l
    )"

    echo "trace_count=$TRACE_COUNT" >> "$METHOD_META"

    if [ "$TRACE_COUNT" -ne "$N_TEST" ]; then
        echo "ERROR: $METHOD has $TRACE_COUNT traces; expected $N_TEST."
        echo "status=failed_trace_count" >> "$METHOD_META"
        exit 1
    fi

    # Basic per-file execution-accounting audit.
    METHOD="$METHOD" TRACE="$TRACE" \
    START_SEED="$START_SEED" END_SEED="$END_SEED" \
    python - <<'PY2'
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
        print("MISSING", p)
        overall = False
        continue

    with np.load(p, allow_pickle=True) as d:
        planned = np.asarray(
            d["planned_steps_per_query"],
            dtype=np.int64,
        )
        executed = np.asarray(
            d["executed_steps_per_query"],
            dtype=np.int64,
        )
        actual = np.asarray(
            d["actual_executed_action"]
        )
        actual_steps = int(
            d["actual_executed_steps"]
        )
        unexecuted = int(
            d["unexecuted_action_steps_total"]
        )

        ok = (
            len(planned) == len(executed)
            and np.all(executed >= 0)
            and np.all(executed <= planned)
            and int(executed.sum()) == actual_steps
            and actual.shape[0] == actual_steps
            and int(planned.sum() - executed.sum())
                == unexecuted
        )

        if not ok:
            print(
                f"seed={seed} execution accounting FAIL"
            )
            overall = False

if not overall:
    raise SystemExit(1)

print(
    f"{method}: validated 50 actual-execution traces."
)
PY2

    echo "status=completed" >> "$METHOD_META"

    echo
    echo "$METHOD COMPLETE"
}

# Locked method order.
run_method original
run_method bsop

cat >> "$META_ROOT/master_run_metadata.txt" <<EOF
status=completed
EOF

echo
echo "===== BUILD ARTIFACT MANIFEST ====="

MANIFEST="$META_ROOT/all_artifact_sha256.txt"

{
    sha256sum "$RUNNER"
    sha256sum "$CKPT"
    sha256sum "$0"

    find "$RESULT_ROOT"         -type f -print0         | sort -z         | xargs -0 sha256sum

    find "$TRACE_ROOT"         -type f -print0         | sort -z         | xargs -0 sha256sum

    find "$META_ROOT"         -maxdepth 1         -type f         ! -name 'all_artifact_sha256.txt'         ! -name 'artifact_verification_final.txt'         -print0         | sort -z         | xargs -0 sha256sum
} > "$MANIFEST"

VERIFY="$META_ROOT/artifact_verification_final.txt"

sha256sum -c "$MANIFEST" > "$VERIFY"

FAILURES="$(
    grep -vc ': OK$' "$VERIFY" || true
)"

if [ "$FAILURES" -ne 0 ]; then
    echo "ERROR: artifact verification failures=$FAILURES"
    exit 1
fi

echo
echo "============================================================"
echo "PUSHT IMAGE BSOP CONFIRMATORY FRESH-50 COMPLETE"
echo "Seeds: $START_SEED-$END_SEED"
echo "Runner SHA: $EXPECTED_RUNNER_SHA"
echo "Checkpoint SHA: $EXPECTED_CHECKPOINT_SHA"
echo "Manifest: $MANIFEST"
echo "============================================================"
