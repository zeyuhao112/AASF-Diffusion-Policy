#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Load Conda when available. The current project was validated in base.
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline

CKPT="data/checkpoints/block_pushing/epoch=2250-test_mean_score=0.287.ckpt"
FROZEN_ORIGINAL="paper_exp/results/blockpush_original_50/eval_log.json"
FROZEN_AASF="paper_exp/results/blockpush_aasf_50/eval_log.json"

for required_file in "$CKPT" "$FROZEN_ORIGINAL" "$FROZEN_AASF"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: missing required file: $required_file" >&2
        exit 2
    fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
REPLAY_ROOT="paper_exp/diagnostics/blockpush_fullbatch_video_${STAMP}"
export REPLAY_ROOT
mkdir -p "$REPLAY_ROOT"

echo "Replay root: $REPLAY_ROOT"
echo "Checkpoint : $CKPT"
echo

run_method() {
    local method="$1"
    local output_dir="$REPLAY_ROOT/$method"
    local log_file="$REPLAY_ROOT/${method}.log"

    # Preserve the exact 50-environment evaluation batch.
    export DP_TEST_START_SEED=4300000
    export DP_N_TEST=50

    # Save videos for seeds 4300000..4300038.
    export DP_N_TEST_VIS=39
    export DP_N_TRAIN=0
    export DP_N_TRAIN_VIS=0

    # Do not alter the evaluator's normal parallel environment count.
    unset DP_N_ENVS || true

    # Remove diagnostics that could change behavior.
    unset DP_TRACE_ENABLE || true
    unset DP_TRACE_DIR || true
    unset DP_ACTION_SMOOTH_THRESHOLD || true
    unset DP_ACTION_SMA_WINDOW || true

    if [ "$method" = "original" ]; then
        unset DP_ACTION_SMOOTH_MODE || true
        unset DP_ACTION_SMOOTH_ALPHA || true
        unset DP_ACTION_SMOOTH_ALPHA_MIN || true
        unset DP_ACTION_SMOOTH_ALPHA_MAX || true
        unset DP_ACTION_SMOOTH_GAIN || true
        unset DP_ACTION_SMOOTH_TAU || true
        unset DP_ACTION_THRESHOLD_RATIO || true
        unset DP_ACTION_MEDIAN_STEP_NORM || true
        unset DP_AASF_DEBUG || true
        unset DP_AASF_DEBUG_LIMIT || true
    elif [ "$method" = "aasf" ]; then
        export DP_ACTION_SMOOTH_MODE=adaptive
        export DP_ACTION_SMOOTH_ALPHA=0.8
        export DP_ACTION_SMOOTH_ALPHA_MIN=0.45
        export DP_ACTION_SMOOTH_ALPHA_MAX=0.8
        export DP_ACTION_SMOOTH_GAIN=0.25
        export DP_ACTION_SMOOTH_TAU=0.0198151707
        export DP_ACTION_THRESHOLD_RATIO=1.2403473459
        export DP_ACTION_MEDIAN_STEP_NORM=0.0159755014
        export DP_AASF_DEBUG=0
    else
        echo "ERROR: unknown method: $method" >&2
        exit 3
    fi

    echo
    echo "========================================"
    echo "Method : $method"
    echo "Output : $output_dir"
    echo "Log    : $log_file"
    echo "========================================"

    set +e
    python -u eval.py \
        --checkpoint "$CKPT" \
        --output_dir "$output_dir" \
        --device cuda:0 \
        2>&1 | tee "$log_file"
    eval_code=${PIPESTATUS[0]}
    set -e

    if [ "$eval_code" -ne 0 ]; then
        echo "ERROR: $method evaluation failed with exit code $eval_code." >&2
        exit "$eval_code"
    fi
}

run_method original
run_method aasf

python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["REPLAY_ROOT"])

frozen_paths = {
    "original": Path("paper_exp/results/blockpush_original_50/eval_log.json"),
    "aasf": Path("paper_exp/results/blockpush_aasf_50/eval_log.json"),
}
replay_paths = {
    method: root / method / "eval_log.json"
    for method in ("original", "aasf")
}
target_seeds = [4300024, 4300038, 4300035]

def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

frozen = {method: load(path) for method, path in frozen_paths.items()}
replay = {method: load(path) for method, path in replay_paths.items()}

mismatches = []

print("\n===== FULL-BATCH REPLAY AUDIT =====")
print(f"{'Seed':<10}{'Method':<12}{'Frozen':<12}{'Replay':<12}Status")

for seed in range(4300000, 4300050):
    key = f"test/sim_max_reward_{seed}"

    for method in ("original", "aasf"):
        if key not in frozen[method]:
            raise KeyError(f"Frozen result missing: {method} {key}")
        if key not in replay[method]:
            raise KeyError(f"Replay result missing: {method} {key}")

        frozen_score = float(frozen[method][key])
        replay_score = float(replay[method][key])
        matched = math.isclose(
            frozen_score,
            replay_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        if not matched:
            mismatches.append((seed, method, frozen_score, replay_score))

        if seed in target_seeds:
            print(
                f"{seed:<10}{method:<12}"
                f"{frozen_score:<12.2f}{replay_score:<12.2f}"
                f"{'PASS' if matched else 'MISMATCH'}"
            )

print()
print("Total score comparisons :", 100)
print("Mismatch count          :", len(mismatches))

if mismatches:
    print("\n===== FIRST MISMATCHES =====")
    for seed, method, frozen_score, replay_score in mismatches[:20]:
        print(f"{seed} {method}: {frozen_score:.2f} -> {replay_score:.2f}")

    raise RuntimeError(
        "Full-batch replay does not match the frozen results. "
        "Do not use these videos for transition analysis."
    )

print("\n===== TARGET VIDEO PATHS =====")
for seed in target_seeds:
    video_key = f"test/sim_video_{seed}"
    print(f"\nSeed {seed}")

    for method in ("original", "aasf"):
        video_path = replay[method].get(video_key)
        if video_path is None:
            raise RuntimeError(f"Missing video key: {method} {video_key}")
        print(f"  {method:<8}: {video_path}")

print()
print("FULL-BATCH VIDEO REPLAY: PASS")
print("Replay directory:", root)
PY

echo
echo "BLOCKPUSH FULL-BATCH FORENSICS: PASS"
echo "Results saved under: $REPLAY_ROOT"
