#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/projects/diffusion_policy"
cd "$PROJECT_ROOT"

TAG="bsop-aasf-confirmatory-frozen-20260805"

PROTOCOL_PATH="paper_artifacts/code/bsop_aasf_confirmatory_protocol_locked.md"
CHECKPOINT="data/checkpoints/pusht.ckpt"

EXPECTED_PROTOCOL_SHA="be0caf893b25ac8da7f8a476ab1833aec6170e830a30c6764d44c1c3bc5be2dd"
EXPECTED_CHECKPOINT_SHA="f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876"

START_SEED=4300100
END_SEED=4300149
N_TEST=50
POLICY_SEED_SALT=20260710

RUN_NAME="bsop_aasf_confirmatory_fresh50"

TRACE_BASE="$PROJECT_ROOT/paper_artifacts/traces/$RUN_NAME"
RESULT_BASE="$PROJECT_ROOT/paper_exp/results/$RUN_NAME"
META_BASE="$PROJECT_ROOT/paper_exp/metadata/$RUN_NAME"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "===== CONFIRMATORY PREFLIGHT ====="

HEAD_COMMIT="$(git rev-parse HEAD)"
TAG_COMMIT="$(git rev-list -n 1 "$TAG")"

echo "HEAD=$HEAD_COMMIT"
echo "TAG =$TAG_COMMIT"

test "$HEAD_COMMIT" = "$TAG_COMMIT" \
    || fail "HEAD does not match frozen tag."

git update-index --refresh >/dev/null 2>&1 || true

TRACKED_STATUS="$(
    git status --porcelain --untracked-files=no
)"

if [ -n "$TRACKED_STATUS" ]; then
    printf '%s\n' "$TRACKED_STATUS" >&2
    fail "Tracked working tree is not clean."
fi

echo \
    "$EXPECTED_PROTOCOL_SHA  $PROTOCOL_PATH" \
    | sha256sum -c -

echo \
    "$EXPECTED_CHECKPOINT_SHA  $CHECKPOINT" \
    | sha256sum -c -

"$PYTHON_BIN" -m py_compile \
    diffusion_policy/env_runner/pusht_keypoints_runner.py \
    diffusion_policy/gym_util/async_vector_env.py \
    diffusion_policy/policy/diffusion_unet_lowdim_policy.py \
    eval.py

echo "CONFIRMATORY PREFLIGHT OK"

for output_path in "$TRACE_BASE" "$RESULT_BASE" "$META_BASE"; do
    if [ -e "$output_path" ]; then
        fail "Output already exists: $output_path"
    fi
done

mkdir -p \
    "$TRACE_BASE" \
    "$RESULT_BASE" \
    "$META_BASE"

MASTER_META="$META_BASE/master_run_metadata.txt"

{
    echo "run_name=$RUN_NAME"
    echo "status=started"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "frozen_tag=$TAG"
    echo "frozen_commit=$TAG_COMMIT"
    echo "head_commit=$HEAD_COMMIT"
    echo "start_seed=$START_SEED"
    echo "end_seed=$END_SEED"
    echo "n_test=$N_TEST"
    echo "n_envs=1"
    echo "policy_seed_salt=$POLICY_SEED_SALT"
    echo "checkpoint=$CHECKPOINT"
    echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"
    echo "protocol=$PROTOCOL_PATH"
    echo "protocol_sha256=$EXPECTED_PROTOCOL_SHA"
    echo "python_bin=$PYTHON_BIN"

    echo
    echo "[Frozen file SHA256]"
    sha256sum \
        diffusion_policy/env_runner/pusht_keypoints_runner.py \
        diffusion_policy/gym_util/async_vector_env.py \
        diffusion_policy/policy/diffusion_unet_lowdim_policy.py \
        eval.py \
        "$PROTOCOL_PATH" \
        "$CHECKPOINT"

    echo
    echo "[Python]"
    "$PYTHON_BIN" --version 2>&1

    echo
    echo "[Package versions]"
    "$PYTHON_BIN" - <<'PYVERSIONS'
import numpy
import torch

print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)

try:
    import diffusers
    print("diffusers=" + diffusers.__version__)
except Exception as exc:
    print("diffusers_error=" + repr(exc))

try:
    import gym
    print("gym=" + gym.__version__)
except Exception as exc:
    print("gym_error=" + repr(exc))
PYVERSIONS

    echo
    echo "[GPU]"
    nvidia-smi || true
} > "$MASTER_META"


validate_method() {
    local trace_dir="$1"
    local method="$2"

    "$PYTHON_BIN" - \
        "$trace_dir" \
        "$START_SEED" \
        "$END_SEED" \
        "$method" <<'PYVALIDATE'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
start_seed = int(sys.argv[2])
end_seed = int(sys.argv[3])
method = sys.argv[4]

expected = set(range(start_seed, end_seed + 1))
normal = {}
diagnostic = {}

for path in root.glob("*.npz"):
    with np.load(path, allow_pickle=False) as data:
        seed = int(data["seed"])

        if path.name.endswith("_overlap_diag.npz"):
            if seed in diagnostic:
                raise RuntimeError(
                    f"{method}: duplicate diagnostic seed {seed}"
                )
            diagnostic[seed] = path
            continue

        if seed in normal:
            raise RuntimeError(
                f"{method}: duplicate normal seed {seed}"
            )

        normal[seed] = path

        if (
            str(data["execution_trace_version"])
            != "actual_execution_trace_v1"
        ):
            raise RuntimeError(
                f"{path}: invalid execution trace version"
            )

        executed_steps = int(
            data["actual_executed_steps"]
        )

        if data["executed_action"].shape[0] != executed_steps:
            raise RuntimeError(
                f"{path}: executed-action count mismatch"
            )

        if data["reward_history"].shape[0] != executed_steps:
            raise RuntimeError(
                f"{path}: reward-history count mismatch"
            )

        planned = np.asarray(
            data["planned_steps_per_query"],
            dtype=np.int64,
        )
        executed = np.asarray(
            data["executed_steps_per_query"],
            dtype=np.int64,
        )

        if planned.shape != executed.shape:
            raise RuntimeError(
                f"{path}: planned/executed shape mismatch"
            )

        if int(executed.sum()) != executed_steps:
            raise RuntimeError(
                f"{path}: executed-step sum mismatch"
            )

if set(normal) != expected:
    raise RuntimeError(
        f"{method}: normal seed mismatch. "
        f"Found {sorted(normal)}"
    )

if set(diagnostic) != expected:
    raise RuntimeError(
        f"{method}: diagnostic seed mismatch. "
        f"Found {sorted(diagnostic)}"
    )

print(
    f"{method}: validated "
    f"{len(normal)} normal and "
    f"{len(diagnostic)} diagnostic traces."
)
PYVALIDATE
}


run_method() {
    local method="$1"

    local trace_dir="$TRACE_BASE/$method"
    local result_dir="$RESULT_BASE/$method"
    local method_meta="$META_BASE/${method}_run_metadata.txt"

    mkdir -p "$trace_dir"

    (
        set -euo pipefail

        export DP_N_ENVS=1
        export DP_N_TEST="$N_TEST"
        export DP_N_TEST_VIS=0
        export DP_N_TRAIN=0
        export DP_N_TRAIN_VIS=0

        export DP_TEST_START_SEED="$START_SEED"
        export DP_PAIRED_POLICY_SEED=1
        export DP_POLICY_SEED_SALT="$POLICY_SEED_SALT"

        export DP_TRACE_ENABLE=1
        export DP_TRACE_DIR="$trace_dir"
        export DP_TRACE_WARMUP_QUERIES=3

        # Uniform diagnostic instrumentation for all methods.
        # These runs are not used for runtime comparisons.
        export DP_OVERLAP_DIAG_ENABLE=1

        export WANDB_MODE=offline
        export SDL_VIDEODRIVER=dummy

        export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
        export DP_ACTION_SMOOTH_ALPHA_MAX=0.8
        export DP_ACTION_SMOOTH_THRESHOLD=10.0
        export DP_ACTION_SMOOTH_GAIN=0.25

        case "$method" in
            original)
                export DP_ACTION_SMOOTH_MODE=fixed
                export DP_ACTION_SMOOTH_ALPHA=1.0

                unset DP_BSOP_ENABLE
                unset DP_BSOP_ETA
                unset DP_BSOP_LAMBDA_MAX
                unset DP_BSOP_RHO
                unset DP_BSOP_KAPPA

                export DP_TRACE_METHOD="confirmatory_fresh50_original"
                ;;

            aasf)
                export DP_ACTION_SMOOTH_MODE=adaptive
                export DP_ACTION_SMOOTH_ALPHA=0.8

                unset DP_BSOP_ENABLE
                unset DP_BSOP_ETA
                unset DP_BSOP_LAMBDA_MAX
                unset DP_BSOP_RHO
                unset DP_BSOP_KAPPA

                export DP_TRACE_METHOD="confirmatory_fresh50_aasf"
                ;;

            bsop)
                export DP_ACTION_SMOOTH_MODE=fixed
                export DP_ACTION_SMOOTH_ALPHA=1.0

                export DP_BSOP_ENABLE=1
                export DP_BSOP_ETA=1.0
                export DP_BSOP_LAMBDA_MAX=0.5
                export DP_BSOP_RHO=5.0
                export DP_BSOP_KAPPA=0.5

                export DP_TRACE_METHOD="confirmatory_fresh50_bsop"
                ;;

            bsop_aasf)
                export DP_ACTION_SMOOTH_MODE=adaptive
                export DP_ACTION_SMOOTH_ALPHA=0.8

                export DP_BSOP_ENABLE=1
                export DP_BSOP_ETA=1.0
                export DP_BSOP_LAMBDA_MAX=0.5
                export DP_BSOP_RHO=5.0
                export DP_BSOP_KAPPA=0.5

                export DP_TRACE_METHOD="confirmatory_fresh50_bsop_aasf"
                ;;

            *)
                fail "Unknown method: $method"
                ;;
        esac

        {
            echo "method=$method"
            echo "status=started"
            echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "frozen_tag=$TAG"
            echo "frozen_commit=$TAG_COMMIT"
            echo "start_seed=$START_SEED"
            echo "end_seed=$END_SEED"
            echo "n_test=$N_TEST"
            echo "checkpoint=$CHECKPOINT"
            echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"

            echo
            echo "[Environment variables]"
            env \
                | grep -E \
                  '^(DP_|WANDB_MODE=|SDL_VIDEODRIVER=)' \
                | sort
        } > "$method_meta"

        echo
        echo "============================================================"
        echo "CONFIRMATORY METHOD: $method"
        echo "SEEDS: $START_SEED-$END_SEED"
        echo "============================================================"

        "$PYTHON_BIN" eval.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$result_dir" \
            --device cuda:0

        validate_method "$trace_dir" "$method"

        normal_count="$(
            find "$trace_dir" \
                -maxdepth 1 \
                -type f \
                -name '*.npz' \
                ! -name '*_overlap_diag.npz' \
                | wc -l
        )"

        diagnostic_count="$(
            find "$trace_dir" \
                -maxdepth 1 \
                -type f \
                -name '*_overlap_diag.npz' \
                | wc -l
        )"

        {
            echo
            echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "status=completed"
            echo "normal_trace_count=$normal_count"
            echo "diagnostic_trace_count=$diagnostic_count"

            echo
            echo "[Trace SHA256]"
            find "$trace_dir" \
                -maxdepth 1 \
                -type f \
                -print0 \
                | sort -z \
                | xargs -0 sha256sum
        } >> "$method_meta"
    )
}


# Locked confirmatory order.
run_method original
run_method aasf
run_method bsop
run_method bsop_aasf

MANIFEST="$META_BASE/all_artifact_sha256.txt"

{
    echo
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "status=completed"
    echo "artifact_manifest=$MANIFEST"
} >> "$MASTER_META"

find "$TRACE_BASE" "$RESULT_BASE" "$META_BASE" \
    -type f \
    ! -path "$MANIFEST" \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$MANIFEST"

echo
echo "============================================================"
echo "ALL FOUR CONFIRMATORY METHODS COMPLETED"
echo "Frozen commit: $TAG_COMMIT"
echo "Seeds: $START_SEED-$END_SEED"
echo "Metadata: $META_BASE"
echo "============================================================"
