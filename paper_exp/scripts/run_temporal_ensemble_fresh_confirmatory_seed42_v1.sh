#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/projects/diffusion_policy

STUDY_ROOT="paper_exp/temporal_ensemble_fresh_seed42"

RUN_ROOT="$STUDY_ROOT/formal_confirmatory"
ANALYSIS_ROOT="$STUDY_ROOT/formal_confirmatory_analysis"

RESULT_ROOT="$RUN_ROOT/results"
TRACE_ROOT="$RUN_ROOT/traces"
LOG_ROOT="$RUN_ROOT/logs"
META_ROOT="$RUN_ROOT/metadata"

START_MARKER="$RUN_ROOT/TE_FRESH_FORMAL_CONFIRMATORY_STARTED.txt"
POSTRUN_FREEZE="$RUN_ROOT/TE_FRESH_CONFIRMATORY_POSTRUN_FREEZE.txt"
MANIFEST="$RUN_ROOT/TE_FRESH_CONFIRMATORY_ARTIFACT_SHA256.txt"

PROTOCOL="$STUDY_ROOT/TE_FRESH_PROTOCOL_FREEZE.txt"
PROTOCOL_MAN="$STUDY_ROOT/TE_FRESH_PROTOCOL_FREEZE_SHA256.txt"

GRID_MAN="$STUDY_ROOT/development_grid/TE_FRESH_DEVELOPMENT_GRID_SHA256.txt"

SELECTION_ROOT="$STUDY_ROOT/development_selection"
SELECTION_MAN="$SELECTION_ROOT/selection_artifact_sha256.txt"
SELECTED="$SELECTION_ROOT/selected_temporal_ensemble_beta.txt"

SELECTION_FREEZE="$STUDY_ROOT/development_selection_freeze_001/TE_FRESH_DEVELOPMENT_SELECTION_FREEZE.txt"
SELECTION_FREEZE_MAN="$STUDY_ROOT/development_selection_freeze_001/TE_FRESH_DEVELOPMENT_SELECTION_FREEZE_SHA256.txt"

BUILD_MAN="$STUDY_ROOT/formal_confirmatory_build_001/TE_FRESH_FORMAL_CONFIRMATORY_BUILD_SHA256.txt"

LAUNCHER="paper_exp/scripts/run_temporal_ensemble_fresh_confirmatory_seed42_v1.sh"

RUNNER="diffusion_policy/env_runner/pusht_keypoints_runner.py"
TE_CORE="diffusion_policy/common/temporal_ensemble.py"
TE_TEST="paper_exp/scripts/test_temporal_ensemble_alignment.py"

CHECKPOINT="data/checkpoints/pusht.ckpt"

START_SEED=4800100
END_SEED=4800149
N_TEST=50

TE_BETA="2.0"
TE_MAX_HISTORY=1

EXPECTED_NORMAL_PER_ARM=50
EXPECTED_DIAG_PER_ARM=0


sha_of() {
    sha256sum "$1" | awk '{print $1}'
}


confirmatory_artifact_count() {
python - <<'PY'
from pathlib import Path
import re

root = Path(
    "paper_exp/temporal_ensemble_fresh_seed42"
)

hits = []

for path in root.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in (
        ".npz",
        ".json",
    ):
        continue

    for token in re.findall(
        r"(?<!\d)(\d{7})(?!\d)",
        path.name,
    ):
        seed = int(token)

        if 4800100 <= seed <= 4800149:
            hits.append(path)
            break

print(len(hits))
PY
}


echo "===== FRESH TE FORMAL CONFIRMATORY PREFLIGHT ====="


echo
echo "===== P1. FROZEN MANIFESTS ====="

sha256sum -c "$PROTOCOL_MAN" >/dev/null 2>&1
echo "TE_FRESH_PROTOCOL_MANIFEST_VERIFY = PASS"

sha256sum -c "$GRID_MAN" >/dev/null 2>&1
echo "DEVELOPMENT_GRID_MANIFEST_VERIFY = PASS"

sha256sum -c "$SELECTION_MAN" >/dev/null 2>&1
echo "SELECTION_ARTIFACT_MANIFEST_VERIFY = PASS"

sha256sum -c "$SELECTION_FREEZE_MAN" >/dev/null 2>&1
echo "DEVELOPMENT_SELECTION_FREEZE_VERIFY = PASS"

sha256sum -c "$BUILD_MAN" >/dev/null 2>&1
echo "FORMAL_CONFIRMATORY_BUILD_MANIFEST_VERIFY = PASS"


echo
echo "===== P2. FROZEN BETA IDENTITY ====="

grep -q '^selected_beta=2$' "$SELECTED"
grep -q '^selected_method=te_beta_2p0$' "$SELECTED"
grep -q '^confirmatory_run_allowed=YES$' "$SELECTED"
grep -q '^confirmatory_seeds_used=NO$' "$SELECTED"

grep -q '^selected_beta = 2.0$' "$SELECTION_FREEZE"
grep -q '^selected_method = te_beta_2p0$' "$SELECTION_FREEZE"
grep -q '^fresh_confirmatory_seeds_used = NO$' "$SELECTION_FREEZE"

echo "SELECTED_BETA = 2.0"
echo "SELECTED_METHOD = te_beta_2p0"
echo "FROZEN_SELECTION_GATE = PASS"


echo
echo "===== P3. CODE / ALIGNMENT ====="

python -m py_compile \
    "$RUNNER" \
    "$TE_CORE" \
    "$TE_TEST"

python "$TE_TEST"

echo "TE_SYNTHETIC_ALIGNMENT_RETEST = PASS"


echo
echo "===== P4. FORMAL OUTPUT MUST BE ABSENT ====="

if [ -e "$RUN_ROOT" ]; then
    echo "FORMAL_CONFIRMATORY_ROOT_ALREADY_EXISTS = YES"
    echo "REFUSING_TO_OVERWRITE = YES"
    exit 1
fi

if [ -e "$ANALYSIS_ROOT" ]; then
    echo "FORMAL_CONFIRMATORY_ANALYSIS_ALREADY_EXISTS = YES"
    echo "REFUSING_TO_OVERWRITE = YES"
    exit 1
fi

CONFIRMATORY_HITS="$(confirmatory_artifact_count)"

echo "confirmatory_artifact_hits=$CONFIRMATORY_HITS"

if [ "$CONFIRMATORY_HITS" -ne 0 ]; then
    echo "TE_CONFIRMATORY_ACCESS_GUARD = FAIL"
    exit 1
fi

echo "TE_CONFIRMATORY_ACCESS_GUARD = PASS"


echo
echo "===== P5. ARM GATE ====="

if [ "${TE_FRESH_CONFIRMATORY_ARM:-NO}" != "YES" ]; then
    echo "TE_FRESH_CONFIRMATORY_ARM != YES"
    echo "FORMAL_CONFIRMATORY_RUN_NOT_STARTED = YES"
    echo "FRESH_TE_CONFIRMATORY_SEEDS_USED = NO"
    echo "TE_CONFIRMATORY_UNBLINDED = NO"
    exit 2
fi


echo
echo "===== FORMAL CONFIRMATORY UNBLINDING START ====="

mkdir -p \
    "$RUN_ROOT" \
    "$RESULT_ROOT" \
    "$TRACE_ROOT" \
    "$LOG_ROOT" \
    "$META_ROOT"

cat > "$START_MARKER" <<EOF2
study=temporal_ensemble_fresh_formal_confirmatory_seed42
status=STARTED
formal_confirmatory_unblinded=YES
start_seed=$START_SEED
end_seed=$END_SEED
n_test=$N_TEST
arms=2
reference=original
selected_method=te_beta_2p0
selected_beta=2.0
alternative_beta_evaluation=FORBIDDEN
post_unblind_retuning=FORBIDDEN
EOF2

echo "FORMAL_CONFIRMATORY_UNBLINDED = YES"
echo "seeds=$START_SEED-$END_SEED"
echo "arms=original,te_beta_2p0"


# ============================================================
# Frozen runtime environment
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

export DP_BSOP_ENABLE=0

export DP_ACTION_SMOOTH_MODE=fixed
export DP_ACTION_SMOOTH_ALPHA=1.0

export DP_TRACE_ENABLE=1
export DP_TRACE_WARMUP_QUERIES=3

# Fresh development did not enable overlap diagnostics.
# Formal confirmatory preserves that frozen instrumentation state.
unset DP_OVERLAP_DIAG_ENABLE || true

export DP_TE_MAX_HISTORY="$TE_MAX_HISTORY"


write_metadata_header() {
    local method="$1"
    local te_enable="$2"
    local beta="$3"
    local trace_dir="$4"
    local meta="$5"

    {
        echo "study=temporal_ensemble_fresh_formal_confirmatory_seed42"
        echo "method=$method"
        echo "status=started"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo

        echo "checkpoint=$CHECKPOINT"
        echo "checkpoint_sha256=$(sha_of "$CHECKPOINT")"
        echo

        echo "start_seed=$START_SEED"
        echo "end_seed=$END_SEED"
        echo "n_test=$N_TEST"
        echo "n_envs=1"
        echo

        echo "DP_TE_ENABLE=$te_enable"
        echo "DP_TE_BETA=$beta"
        echo "DP_TE_MAX_HISTORY=$TE_MAX_HISTORY"
        echo

        echo "DP_PAIRED_POLICY_SEED=$DP_PAIRED_POLICY_SEED"
        echo "DP_POLICY_SEED_SALT=$DP_POLICY_SEED_SALT"
        echo

        echo "DP_BSOP_ENABLE=$DP_BSOP_ENABLE"
        echo "DP_ACTION_SMOOTH_MODE=$DP_ACTION_SMOOTH_MODE"
        echo "DP_ACTION_SMOOTH_ALPHA=$DP_ACTION_SMOOTH_ALPHA"
        echo

        echo "DP_TRACE_ENABLE=$DP_TRACE_ENABLE"
        echo "DP_OVERLAP_DIAG_ENABLE=NOT_SET"
        echo "DP_TRACE_DIR=$PWD/$trace_dir"
        echo

        echo "protocol_sha256=$(sha_of "$PROTOCOL")"
        echo "selected_beta_file_sha256=$(sha_of "$SELECTED")"
        echo "selection_freeze_sha256=$(sha_of "$SELECTION_FREEZE")"

        echo "runner_sha256=$(sha_of "$RUNNER")"
        echo "te_core_sha256=$(sha_of "$TE_CORE")"
        echo "te_test_sha256=$(sha_of "$TE_TEST")"
        echo "launcher_sha256=$(sha_of "$LAUNCHER")"
        echo
    } > "$meta"
}


finish_metadata() {
    local trace_dir="$1"
    local meta="$2"

    local normal_count
    local diag_count

    normal_count="$(
        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name 'test_seed_*.npz' \
            ! -name '*_overlap_diag.npz' \
            | wc -l
    )"

    diag_count="$(
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
        echo "diagnostic_trace_count=$diag_count"
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
}


exact_seed_audit() {
    local method="$1"
    local trace_dir="$2"

    METHOD="$method" \
    TRACE_DIR="$trace_dir" \
    START_SEED="$START_SEED" \
    END_SEED="$END_SEED" \
    python - <<'PY'
from pathlib import Path
import os
import re

method = os.environ["METHOD"]
root = Path(os.environ["TRACE_DIR"])
start = int(os.environ["START_SEED"])
end = int(os.environ["END_SEED"])

expected = set(range(start, end + 1))
observed = []

for p in root.glob("test_seed_*.npz"):

    if p.name.endswith("_overlap_diag.npz"):
        continue

    m = re.fullmatch(
        r"test_seed_(\d+)\.npz",
        p.name,
    )

    if m is None:
        raise RuntimeError(
            "Unexpected trace filename: {}".format(p)
        )

    observed.append(int(m.group(1)))

if len(observed) != len(set(observed)):
    raise RuntimeError(
        "{}: duplicate trace seeds".format(method)
    )

if set(observed) != expected:
    raise RuntimeError(
        "{}: exact formal seed set mismatch".format(method)
    )

print(
    "{}_EXACT_FORMAL_SEED_SET = PASS".format(method)
)
PY
}


run_method() {
    local method="$1"
    local te_enable="$2"
    local beta="$3"

    local result_dir="$RESULT_ROOT/$method"
    local trace_dir="$TRACE_ROOT/$method"
    local log="$LOG_ROOT/$method.log"
    local meta="$META_ROOT/$method.txt"

    echo
    echo "============================================================"
    echo "FORMAL CONFIRMATORY METHOD: $method"
    echo "============================================================"

    for path in \
        "$result_dir" \
        "$trace_dir" \
        "$log" \
        "$meta"
    do
        if [ -e "$path" ]; then
            echo "FORMAL_ARM_ARTIFACT_ALREADY_EXISTS = $path"
            echo "REFUSING_TO_OVERWRITE = YES"
            exit 1
        fi
    done

    mkdir "$trace_dir"

    export DP_TE_ENABLE="$te_enable"
    export DP_TE_BETA="$beta"

    export DP_TRACE_DIR="$PWD/$trace_dir"
    export DP_TRACE_METHOD="te_fresh_confirmatory_${method}"

    write_metadata_header \
        "$method" \
        "$te_enable" \
        "$beta" \
        "$trace_dir" \
        "$meta"

    set +e

    python eval.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$result_dir" \
        --device cuda:0 \
        2>&1 | tee "$log"

    status=${PIPESTATUS[0]}

    set -e

    if [ "$status" -ne 0 ]; then
        {
            echo
            echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "status=FAILED"
            echo "python_exit_status=$status"
        } >> "$meta"

        echo "METHOD_FAILED = $method"
        exit "$status"
    fi

    test -s "$result_dir/eval_log.json"

    normal_count="$(
        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name 'test_seed_*.npz' \
            ! -name '*_overlap_diag.npz' \
            | wc -l
    )"

    diag_count="$(
        find "$trace_dir" \
            -maxdepth 1 \
            -type f \
            -name '*_overlap_diag.npz' \
            | wc -l
    )"

    echo "normal_trace_count=$normal_count"
    echo "diagnostic_trace_count=$diag_count"

    if [ "$normal_count" -ne "$EXPECTED_NORMAL_PER_ARM" ]; then
        echo "NORMAL_TRACE_COUNT_FAILURE = YES"
        exit 1
    fi

    if [ "$diag_count" -ne "$EXPECTED_DIAG_PER_ARM" ]; then
        echo "DIAGNOSTIC_TRACE_COUNT_FAILURE = YES"
        exit 1
    fi

    if grep -nEi \
        'Traceback|RuntimeError|Negative query age|No aligned prediction|alignment.*failed|cursor.*failed|CUDA out of memory|Killed' \
        "$log"
    then
        echo "ERROR_SCAN_FAILURE = $method"
        exit 1
    fi

    marker_count="$(
        grep -c \
            '\[temporal-ensemble\]' \
            "$log" \
            || true
    )"

    if [ "$te_enable" -eq 1 ]; then

        echo "temporal_ensemble_marker_count=$marker_count"

        if [ "$marker_count" -ne "$N_TEST" ]; then
            echo "TEMPORAL_ENSEMBLE_MARKER_COUNT_FAILURE = YES"
            exit 1
        fi

        if grep \
            '\[temporal-ensemble\]' \
            "$log" \
            | grep -v \
                'beta=2,'
        then
            echo "UNEXPECTED_BETA_IN_RUNTIME_MARKER = YES"
            exit 1
        fi

        if grep \
            '\[temporal-ensemble\]' \
            "$log" \
            | grep -v \
                'actual_cursor=PASS'
        then
            echo "ACTUAL_CURSOR_RUNTIME_FAILURE = YES"
            exit 1
        fi

    else

        if [ "$marker_count" -ne 0 ]; then
            echo "ORIGINAL_RUN_CONTAINS_TE_MARKERS = YES"
            exit 1
        fi
    fi

    exact_seed_audit \
        "$method" \
        "$trace_dir"

    finish_metadata \
        "$trace_dir" \
        "$meta"

    echo "METHOD_COMPLETE = $method"
}


# ============================================================
# Exactly two frozen formal arms.
# No alternative beta may be evaluated.
# ============================================================

run_method \
    "original" \
    0 \
    "0.0"

run_method \
    "te_beta_2p0" \
    1 \
    "$TE_BETA"


echo
echo "===== FORMAL CONFIRMATORY COMPLETION AUDIT ====="

for method in \
    original \
    te_beta_2p0
do
    grep -q \
        '^status=completed$' \
        "$META_ROOT/$method.txt"

    test -s \
        "$RESULT_ROOT/$method/eval_log.json"
done


TOTAL_NORMAL="$(
    find "$TRACE_ROOT" \
        -type f \
        -name 'test_seed_*.npz' \
        ! -name '*_overlap_diag.npz' \
        | wc -l
)"

TOTAL_DIAG="$(
    find "$TRACE_ROOT" \
        -type f \
        -name '*_overlap_diag.npz' \
        | wc -l
)"

echo "total_normal_traces=$TOTAL_NORMAL"
echo "total_diagnostic_traces=$TOTAL_DIAG"

if [ "$TOTAL_NORMAL" -ne 100 ]; then
    echo "FORMAL_NORMAL_TRACE_ACCOUNTING = FAIL"
    exit 1
fi

if [ "$TOTAL_DIAG" -ne 0 ]; then
    echo "FORMAL_DIAGNOSTIC_TRACE_ACCOUNTING = FAIL"
    exit 1
fi

echo "FORMAL_NORMAL_TRACE_ACCOUNTING = PASS"
echo "FORMAL_DIAGNOSTIC_TRACE_ACCOUNTING = PASS"


cat > "$POSTRUN_FREEZE" <<EOF2
===== FRESH TE FORMAL CONFIRMATORY POSTRUN FREEZE =====

study = Push-T low-dimensional Temporal Ensemble baseline
training_checkpoint = seed42

formal_seed_start = 4800100
formal_seed_end = 4800149
formal_seed_count = 50

formal_arms = 2
formal_reference = original
formal_selected_method = te_beta_2p0
formal_selected_beta = 2.0

original_episode_count = 50
selected_TE_episode_count = 50
total_episode_count = 100

normal_trace_count = 100
diagnostic_trace_count = 0
DP_OVERLAP_DIAG_ENABLE = NOT_SET

DP_TE_MAX_HISTORY = 1
DP_N_ENVS = 1
DP_PAIRED_POLICY_SEED = 1
DP_POLICY_SEED_SALT = 20260710

development_beta_selection_performed = YES
alternative_beta_evaluation_performed = NO
post_unblind_retuning = NO
new_hyperparameter_search_after_unblinding = NO

formal_confirmatory_unblinded = YES
EOF2


echo
echo "===== FORMAL CONFIRMATORY ARTIFACT MANIFEST ====="

{
    sha256sum \
        "$PROTOCOL" \
        "$PROTOCOL_MAN" \
        "$GRID_MAN" \
        "$SELECTION_MAN" \
        "$SELECTED" \
        "$SELECTION_ROOT/temporal_ensemble_beta_selection.csv" \
        "$SELECTION_FREEZE" \
        "$SELECTION_FREEZE_MAN" \
        "$BUILD_MAN" \
        "$LAUNCHER" \
        "$RUNNER" \
        "$TE_CORE" \
        "$TE_TEST" \
        "$CHECKPOINT" \
        "$START_MARKER" \
        "$POSTRUN_FREEZE" \
        "$LOG_ROOT/original.log" \
        "$LOG_ROOT/te_beta_2p0.log" \
        "$META_ROOT/original.txt" \
        "$META_ROOT/te_beta_2p0.txt" \
        "$RESULT_ROOT/original/eval_log.json" \
        "$RESULT_ROOT/te_beta_2p0/eval_log.json"

    find "$TRACE_ROOT" \
        -type f \
        -name '*.npz' \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum

} > "$MANIFEST"


echo
echo "===== FINAL MANIFEST VERIFY ====="

sha256sum -c "$MANIFEST" >/dev/null 2>&1

echo "TE_FRESH_CONFIRMATORY_MANIFEST_VERIFY = PASS"

sha256sum -c \
    paper_submission_final_frozen_20260816.tar.gz.sha256.txt \
    >/dev/null 2>&1

echo "FROZEN_SUBMISSION_ARCHIVE_POST_CONFIRMATORY = PASS"


echo
echo "===== FINAL FORMAL CONFIRMATORY STATUS ====="

echo "TE_FRESH_FORMAL_CONFIRMATORY_RUN = PASS"
echo "FORMAL_CONFIRMATORY_UNBLINDED = YES"

echo "FORMAL_SEEDS = 4800100-4800149"
echo "FORMAL_ARMS = original,te_beta_2p0"
echo "FORMAL_SELECTED_BETA = 2.0"

echo "TOTAL_FORMAL_EPISODES = 100"
echo "TOTAL_NORMAL_TRACES = 100"
echo "TOTAL_DIAGNOSTIC_TRACES = 0"

echo "ALTERNATIVE_BETA_EVALUATION = NO"
echo "POST_UNBLIND_RETUNING = NO"

echo "READY_FOR_FROZEN_CONFIRMATORY_ANALYSIS = YES"
