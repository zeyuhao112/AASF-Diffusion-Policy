#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/projects/diffusion_policy"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"

RUNNER="diffusion_policy/env_runner/blockpush_lowdim_runner.py"

CHECKPOINT="data/checkpoints/block_pushing/epoch=2250-test_mean_score=0.287.ckpt"

EXPECTED_RUNNER_SHA="f9e27ff74702fdda0741963828cdf03521ac0f901095ca8f1593ac146cf22aad"

EXPECTED_CHECKPOINT_SHA="7318ee33a677fa2682a20b0462bb5639d9105cdfd9d3646ca443931dacd30e0f"

START_SEED=4400100
END_SEED=4400149
N_TEST=50

RUN_NAME="blockpush_bsop_aasf_confirmatory_fresh50"

TRACE_BASE="$PROJECT_ROOT/paper_artifacts/traces/$RUN_NAME"
RESULT_BASE="$PROJECT_ROOT/paper_exp/results/$RUN_NAME"
META_BASE="$PROJECT_ROOT/paper_exp/metadata/$RUN_NAME"

LOG_FILE="$PROJECT_ROOT/paper_exp/logs/${RUN_NAME}.log"

mkdir -p \
    "$PROJECT_ROOT/paper_artifacts/traces" \
    "$PROJECT_ROOT/paper_exp/results" \
    "$PROJECT_ROOT/paper_exp/metadata" \
    "$PROJECT_ROOT/paper_exp/logs"

fail() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}


echo "============================================================"
echo "BLOCKPUSH CONFIRMATORY FRESH-50"
echo "============================================================"


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

test -f "$RUNNER" \
    || fail "Runner missing: $RUNNER"

test -f "$CHECKPOINT" \
    || fail "Checkpoint missing: $CHECKPOINT"


actual_runner_sha="$(
    sha256sum "$RUNNER" | awk '{print $1}'
)"

actual_checkpoint_sha="$(
    sha256sum "$CHECKPOINT" | awk '{print $1}'
)"


if [ "$actual_runner_sha" != "$EXPECTED_RUNNER_SHA" ]; then
    fail \
        "Runner SHA mismatch. Expected $EXPECTED_RUNNER_SHA, got $actual_runner_sha"
fi


if [ "$actual_checkpoint_sha" != "$EXPECTED_CHECKPOINT_SHA" ]; then
    fail \
        "Checkpoint SHA mismatch. Expected $EXPECTED_CHECKPOINT_SHA, got $actual_checkpoint_sha"
fi


"$PYTHON_BIN" -m py_compile "$RUNNER"


# Confirmatory run directories must be fresh.
for path in \
    "$TRACE_BASE" \
    "$RESULT_BASE" \
    "$META_BASE"
do
    if [ -e "$path" ]; then
        fail \
            "Confirmatory target already exists: $path
Refusing to overwrite a locked experiment."
    fi
done


mkdir -p \
    "$TRACE_BASE" \
    "$RESULT_BASE" \
    "$META_BASE"


# ------------------------------------------------------------
# Seed list
# ------------------------------------------------------------

TRACE_SEEDS="$(
    seq "$START_SEED" "$END_SEED" \
    | paste -sd, -
)"


# ------------------------------------------------------------
# Master metadata
# ------------------------------------------------------------

MASTER_META="$META_BASE/master_run_metadata.txt"

{
    echo "run_name=$RUN_NAME"
    echo "status=started"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo
    echo "[Experiment]"
    echo "start_seed=$START_SEED"
    echo "end_seed=$END_SEED"
    echo "n_test=$N_TEST"

    echo
    echo "[Runner]"
    echo "runner=$RUNNER"
    echo "runner_sha256=$EXPECTED_RUNNER_SHA"

    echo
    echo "[Checkpoint]"
    echo "checkpoint=$CHECKPOINT"
    echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"

    echo
    echo "[Paired random numbers]"
    echo "paired_policy_seed=1"
    echo "policy_seed_salt=20260710"

    echo
    echo "[Scale transfer]"
    echo "aasf_tau=0.019815170512094502"
    echo "aasf_alpha=0.8"
    echo "aasf_alpha_min=0.45"
    echo "aasf_alpha_max=0.8"
    echo "aasf_gain=0.25"

    echo
    echo "bsop_eta=1.0"
    echo "bsop_lambda_max=0.5"
    echo "bsop_rho=0.009907585256047251"
    echo "bsop_kappa=0.5"

    echo
    echo "[Methods]"
    echo "original=AASF off, BSOP off"
    echo "aasf=AASF adaptive, BSOP off"
    echo "bsop=AASF off, BSOP on"
    echo "bsop_aasf=AASF adaptive, BSOP on"

    echo
    echo "[Host]"
    uname -a

    echo
    nvidia-smi || true

} > "$MASTER_META"


# ------------------------------------------------------------
# Trace validation
# ------------------------------------------------------------

validate_method() {
    local trace_dir="$1"
    local method="$2"

    METHOD="$method" \
    TRACE_DIR="$trace_dir" \
    START_SEED="$START_SEED" \
    END_SEED="$END_SEED" \
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import numpy as np

method = os.environ["METHOD"]
root = Path(os.environ["TRACE_DIR"])

start_seed = int(os.environ["START_SEED"])
end_seed = int(os.environ["END_SEED"])

expected = set(range(start_seed, end_seed + 1))

paths = sorted(root.glob("*.npz"))

if len(paths) != len(expected):
    raise RuntimeError(
        f"{method}: expected {len(expected)} trace files, "
        f"found {len(paths)}"
    )

found = set()

for path in paths:
    with np.load(path, allow_pickle=True) as data:
        seed = int(data["seed"])

        if seed in found:
            raise RuntimeError(
                f"{method}: duplicate seed {seed}"
            )

        found.add(seed)

        version = str(data["execution_trace_version"])

        if version != "actual_execution_trace_v1":
            raise RuntimeError(
                f"{method} seed {seed}: "
                f"unexpected execution trace version "
                f"{version}"
            )

        planned = np.asarray(
            data["planned_steps_per_query"],
            dtype=np.int64,
        )

        executed = np.asarray(
            data["executed_steps_per_query"],
            dtype=np.int64,
        )

        actual_action = np.asarray(
            data["actual_executed_action"]
        )

        actual_steps = int(
            data["actual_executed_steps"]
        )

        unexecuted = int(
            data["unexecuted_action_steps_total"]
        )

        if len(planned) != len(executed):
            raise RuntimeError(
                f"{method} seed {seed}: "
                "planned/executed query length mismatch"
            )

        if np.any(executed < 0):
            raise RuntimeError(
                f"{method} seed {seed}: "
                "negative executed-step count"
            )

        if np.any(executed > planned):
            raise RuntimeError(
                f"{method} seed {seed}: "
                "executed more than planned"
            )

        if int(executed.sum()) != actual_steps:
            raise RuntimeError(
                f"{method} seed {seed}: "
                "actual-step sum mismatch"
            )

        if actual_action.shape[0] != actual_steps:
            raise RuntimeError(
                f"{method} seed {seed}: "
                "actual action trajectory mismatch"
            )

        expected_unexecuted = (
            int(planned.sum()) - actual_steps
        )

        if unexecuted != expected_unexecuted:
            raise RuntimeError(
                f"{method} seed {seed}: "
                "unexecuted-step mismatch"
            )

        # BSOP diagnostics must obey the safety constraints.
        if method in {"bsop", "bsop_aasf"}:
            required = [
                "bsop_boundary_executed",
                "bsop_raw_boundary_jump",
                "bsop_projected_boundary_jump",
                "bsop_correction",
                "bsop_relative_correction",
                "bsop_active",
                "bsop_rho",
                "bsop_kappa",
            ]

            missing = [
                key for key in required
                if key not in data.files
            ]

            if missing:
                raise RuntimeError(
                    f"{method} seed {seed}: "
                    f"missing BSOP fields {missing}"
                )

            boundary_executed = np.asarray(
                data["bsop_boundary_executed"],
                dtype=bool,
            )

            raw = np.asarray(
                data["bsop_raw_boundary_jump"],
                dtype=np.float64,
            )

            projected = np.asarray(
                data["bsop_projected_boundary_jump"],
                dtype=np.float64,
            )

            correction = np.asarray(
                data["bsop_correction"],
                dtype=np.float64,
            )

            rho = float(data["bsop_rho"])
            kappa = float(data["bsop_kappa"])

            mask = (
                boundary_executed
                & np.isfinite(raw)
                & np.isfinite(projected)
                & np.isfinite(correction)
            )

            if np.any(
                projected[mask]
                > raw[mask] + 1e-4
            ):
                raise RuntimeError(
                    f"{method} seed {seed}: "
                    "boundary safety violation"
                )

            if np.any(
                correction[mask]
                > rho + 1e-4
            ):
                raise RuntimeError(
                    f"{method} seed {seed}: "
                    "absolute correction cap violation"
                )

            if np.any(
                correction[mask]
                > kappa * raw[mask] + 1e-4
            ):
                raise RuntimeError(
                    f"{method} seed {seed}: "
                    "relative correction cap violation"
                )

if found != expected:
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    raise RuntimeError(
        f"{method}: seed mismatch. "
        f"missing={missing}, extra={extra}"
    )

print(
    f"{method}: validated "
    f"{len(found)} fresh-50 traces."
)
PY
}


# ------------------------------------------------------------
# One method
# ------------------------------------------------------------

run_method() {
    local method="$1"

    local trace_dir="$TRACE_BASE/$method"
    local result_dir="$RESULT_BASE/$method"
    local method_meta="$META_BASE/${method}_run_metadata.txt"

    mkdir -p "$trace_dir"

    (
        set -euo pipefail

        # Evaluation geometry.
        export DP_N_ENVS=1
        export DP_N_TEST="$N_TEST"
        export DP_N_TEST_VIS=0
        export DP_N_TRAIN=0
        export DP_N_TRAIN_VIS=0
        export DP_TEST_START_SEED="$START_SEED"

        # Common-random-number protocol.
        export DP_PAIRED_POLICY_SEED=1
        export DP_POLICY_SEED_SALT=20260710

        # Full forensic traces.
        export DP_AASF_TRACE=1
        export DP_AASF_TRACE_DIR="$trace_dir"
        export DP_AASF_TRACE_SEEDS="$TRACE_SEEDS"

        # Keep evaluation headless.
        export WANDB_MODE=offline
        export SDL_VIDEODRIVER=dummy

        # Frozen cross-task AASF parameters.
        export DP_ACTION_SMOOTH_ALPHA=0.8
        export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
        export DP_ACTION_SMOOTH_ALPHA_MAX=0.8
        export DP_ACTION_SMOOTH_GAIN=0.25
        export DP_ACTION_SMOOTH_TAU=0.019815170512094502

        # Frozen cross-task BSOP parameters.
        export DP_BSOP_ETA=1.0
        export DP_BSOP_LAMBDA_MAX=0.5
        export DP_BSOP_RHO=0.009907585256047251
        export DP_BSOP_KAPPA=0.5

        case "$method" in
            original)
                export DP_ACTION_SMOOTH_MODE=off
                export DP_BSOP_ENABLE=0
                ;;

            aasf)
                export DP_ACTION_SMOOTH_MODE=adaptive
                export DP_BSOP_ENABLE=0
                ;;

            bsop)
                export DP_ACTION_SMOOTH_MODE=off
                export DP_BSOP_ENABLE=1
                ;;

            bsop_aasf)
                export DP_ACTION_SMOOTH_MODE=adaptive
                export DP_BSOP_ENABLE=1
                ;;

            *)
                fail "Unknown method: $method"
                ;;
        esac


        {
            echo "method=$method"
            echo "status=started"
            echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

            echo "start_seed=$START_SEED"
            echo "end_seed=$END_SEED"
            echo "n_test=$N_TEST"

            echo "runner=$RUNNER"
            echo "runner_sha256=$EXPECTED_RUNNER_SHA"

            echo "checkpoint=$CHECKPOINT"
            echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"

            echo
            echo "[Environment variables]"

            env \
                | grep '^DP_' \
                | sort

        } > "$method_meta"


        echo
        echo "============================================================"
        echo "BLOCKPUSH METHOD: $method"
        echo "SEEDS: $START_SEED-$END_SEED"
        echo "============================================================"


        "$PYTHON_BIN" eval.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$result_dir" \
            --device cuda:0


        validate_method \
            "$trace_dir" \
            "$method"


        trace_count="$(
            find "$trace_dir" \
                -maxdepth 1 \
                -type f \
                -name '*.npz' \
                | wc -l
        )"


        {
            echo
            echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "status=completed"
            echo "trace_count=$trace_count"

            echo
            echo "[Eval result SHA256]"

            sha256sum \
                "$result_dir/eval_log.json"

            echo
            echo "[Trace SHA256]"

            find "$trace_dir" \
                -maxdepth 1 \
                -type f \
                -name '*.npz' \
                -print0 \
                | sort -z \
                | xargs -0 sha256sum

        } >> "$method_meta"
    )
}


# ------------------------------------------------------------
# Locked confirmatory ordering
# ------------------------------------------------------------

run_method original
run_method aasf
run_method bsop
run_method bsop_aasf


# ------------------------------------------------------------
# Final manifest
# ------------------------------------------------------------

MANIFEST="$META_BASE/all_artifact_sha256.txt"

{
    echo
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "status=completed"
    echo "artifact_manifest=$MANIFEST"
} >> "$MASTER_META"


find \
    "$TRACE_BASE" \
    "$RESULT_BASE" \
    "$META_BASE" \
    -type f \
    ! -path "$MANIFEST" \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$MANIFEST"


echo
echo "============================================================"
echo "ALL FOUR BLOCKPUSH METHODS COMPLETED"
echo "Seeds: $START_SEED-$END_SEED"
echo "Runner SHA: $EXPECTED_RUNNER_SHA"
echo "Checkpoint SHA: $EXPECTED_CHECKPOINT_SHA"
echo "Metadata: $META_BASE"
echo "============================================================"

