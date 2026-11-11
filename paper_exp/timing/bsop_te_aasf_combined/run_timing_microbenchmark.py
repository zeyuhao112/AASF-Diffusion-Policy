import os

# Freeze numerical-library threading before NumPy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import gc
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

# Permanently pin this benchmark to the project tree
# containing the script, rather than relying on shell
# PYTHONPATH or another diffusion_policy clone.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import numpy as np
import diffusion_policy.common.temporal_ensemble as te_module

from diffusion_policy.common.temporal_ensemble import (
    temporal_ensemble_current_chunk,
    append_temporal_ensemble_record,
)


# ============================================================
# Frozen benchmark protocol
# ============================================================

N_BLOCKS = 200
CALLS_PER_BLOCK = 1000
WARMUP_CALLS = 5000

RNG_SEED = 20260828

# Push-T formal setting
H = 16
K = 8
D = 2

# Frozen BSOP parameters
BSOP_ETA = 1.0
BSOP_LAMBDA_MAX = 0.5
BSOP_RHO = 5.0
BSOP_KAPPA = 0.5

# Frozen AASF parameters
AASF_ALPHA_MIN = 0.45
AASF_ALPHA_MAX = 0.80
AASF_THRESHOLD = 10.0
AASF_GAIN = 0.25

# Frozen formal TE setting
TE_BETA = 2.0
TE_MAX_HISTORY = 1


EXPECTED_TE_FILE = (
    PROJECT_ROOT
    / "diffusion_policy"
    / "common"
    / "temporal_ensemble.py"
).resolve()

ACTUAL_TE_FILE = Path(
    te_module.__file__
).resolve()

if ACTUAL_TE_FILE != EXPECTED_TE_FILE:
    raise RuntimeError(
        "Wrong temporal_ensemble implementation: "
        f"{ACTUAL_TE_FILE}; expected {EXPECTED_TE_FILE}"
    )


ROOT = PROJECT_ROOT

OUT_DIR = Path(__file__).resolve().parent

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Machine information
# ============================================================

def read_cpu_model():
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def get_gpu_info():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unavailable"


# Pin to one allowed CPU when possible.
affinity_before = None
affinity_after = None
pin_status = "not attempted"

try:
    affinity_before = sorted(
        os.sched_getaffinity(0)
    )

    if len(affinity_before) > 0:
        cpu = affinity_before[0]
        os.sched_setaffinity(
            0,
            {cpu},
        )
        affinity_after = sorted(
            os.sched_getaffinity(0)
        )
        pin_status = f"pinned to CPU {cpu}"
except Exception as exc:
    pin_status = (
        "pinning unavailable: "
        + repr(exc)
    )


machine_info = {
    "timestamp_unix": time.time(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "cpu_model": read_cpu_model(),
    "logical_cpu_count": os.cpu_count(),
    "affinity_before": affinity_before,
    "affinity_after": affinity_after,
    "pin_status": pin_status,
    "numpy_version": np.__version__,
    "gpu": get_gpu_info(),
    "protocol": {
        "n_blocks": N_BLOCKS,
        "calls_per_block": CALLS_PER_BLOCK,
        "warmup_calls": WARMUP_CALLS,
        "H": H,
        "K": K,
        "D": D,
        "bsop_eta": BSOP_ETA,
        "bsop_lambda_max": BSOP_LAMBDA_MAX,
        "bsop_rho": BSOP_RHO,
        "bsop_kappa": BSOP_KAPPA,
        "aasf_alpha_min": AASF_ALPHA_MIN,
        "aasf_alpha_max": AASF_ALPHA_MAX,
        "aasf_threshold": AASF_THRESHOLD,
        "aasf_gain": AASF_GAIN,
        "te_beta": TE_BETA,
        "te_max_history": TE_MAX_HISTORY,
    },
}


# ============================================================
# Load one real formal Push-T trace
# ============================================================

MAIN_PATH = (
    ROOT
    / "paper_artifacts"
    / "traces"
    / "bsop_aasf_confirmatory_fresh50"
    / "original"
    / "test_seed_4300100.npz"
)

DIAG_PATH = (
    ROOT
    / "paper_artifacts"
    / "traces"
    / "bsop_aasf_confirmatory_fresh50"
    / "original"
    / "test_seed_4300100_overlap_diag.npz"
)


if not MAIN_PATH.exists():
    raise FileNotFoundError(
        f"Missing formal trace: {MAIN_PATH}"
    )

if not DIAG_PATH.exists():
    raise FileNotFoundError(
        f"Missing overlap trace: {DIAG_PATH}"
    )


with np.load(
    MAIN_PATH,
    allow_pickle=False,
) as z:
    executed_action = np.asarray(
        z["executed_action"],
        dtype=np.float32,
    )


with np.load(
    DIAG_PATH,
    allow_pickle=False,
) as z:
    action_pred = np.asarray(
        z["action_pred"],
        dtype=np.float32,
    )

    action_slice_start_arr = np.asarray(
        z["action_slice_start"]
    )

    action_slice_start = int(
        action_slice_start_arr[0]
        if action_slice_start_arr.ndim > 0
        else action_slice_start_arr
    )

    n_action_steps = int(
        np.asarray(
            z["n_action_steps"]
        )
    )


if action_pred.ndim != 3:
    raise RuntimeError(
        "action_pred must be 3D"
    )

if action_pred.shape[1:] != (
    H,
    D,
):
    raise RuntimeError(
        "Unexpected formal action_pred "
        f"shape: {action_pred.shape}"
    )

if n_action_steps != K:
    raise RuntimeError(
        "Unexpected n_action_steps: "
        f"{n_action_steps}"
    )

if action_slice_start != 1:
    raise RuntimeError(
        "Unexpected action_slice_start: "
        f"{action_slice_start}"
    )


# ============================================================
# Exact BSOP core used in runner
# ============================================================

def bsop_exact(
    y,
    a,
    p,
):
    bsop_y = np.asarray(
        y,
        dtype=np.float64,
    )

    bsop_a = np.asarray(
        a,
        dtype=np.float64,
    )

    bsop_p = np.asarray(
        p,
        dtype=np.float64,
    )

    if not (
        bsop_y.shape
        == bsop_a.shape
        == bsop_p.shape
    ):
        raise RuntimeError(
            "BSOP vector shapes differ"
        )

    bsop_d = (
        bsop_a
        - bsop_y
    )

    bsop_q = (
        bsop_p
        - bsop_a
    )

    bsop_q_norm_sq = np.sum(
        bsop_q * bsop_q,
        axis=-1,
    )

    bsop_q_norm = np.sqrt(
        bsop_q_norm_sq
    )

    bsop_raw_jump = np.linalg.norm(
        bsop_d,
        axis=-1,
    )

    bsop_lambda_star = np.zeros_like(
        bsop_q_norm_sq
    )

    bsop_valid = (
        bsop_q_norm_sq
        > 1.0e-12
    )

    bsop_lambda_star[
        bsop_valid
    ] = (
        -np.sum(
            bsop_d[bsop_valid]
            * bsop_q[bsop_valid],
            axis=-1,
        )
        / bsop_q_norm_sq[
            bsop_valid
        ]
    )

    bsop_lambda_star = np.clip(
        bsop_lambda_star,
        0.0,
        1.0,
    )

    bsop_lambda = (
        BSOP_ETA
        * bsop_lambda_star
    )

    bsop_lambda = np.minimum(
        bsop_lambda,
        BSOP_LAMBDA_MAX,
    )

    bsop_lambda = np.minimum(
        bsop_lambda,
        BSOP_RHO
        / (
            bsop_q_norm
            + 1.0e-12
        ),
    )

    # Relative response protection:
    #
    # ||b-a|| <= kappa ||a-y||
    bsop_lambda = np.minimum(
        bsop_lambda,
        BSOP_KAPPA
        * bsop_raw_jump
        / (
            bsop_q_norm
            + 1.0e-12
        ),
    )

    bsop_lambda = np.maximum(
        bsop_lambda,
        0.0,
    )

    bsop_projected_first = (
        bsop_a
        + bsop_lambda[:, None]
        * bsop_q
    )

    bsop_baseline_jump = (
        bsop_raw_jump
    )

    bsop_projected_jump = np.linalg.norm(
        bsop_projected_first
        - bsop_y,
        axis=-1,
    )

    bsop_correction = np.linalg.norm(
        bsop_projected_first
        - bsop_a,
        axis=-1,
    )

    if np.any(
        bsop_projected_jump
        > bsop_baseline_jump
        + 1.0e-8
    ):
        raise RuntimeError(
            "BSOP boundary guarantee "
            "failed before dtype cast."
        )

    if np.any(
        bsop_correction
        > BSOP_RHO
        + 1.0e-8
    ):
        raise RuntimeError(
            "BSOP correction-radius "
            "constraint failed."
        )

    if np.any(
        bsop_correction
        > (
            BSOP_KAPPA
            * bsop_raw_jump
            + 1.0e-8
        )
    ):
        raise RuntimeError(
            "BSOP relative-response "
            "constraint failed."
        )

    return bsop_projected_first


# ============================================================
# Exact AASF numerical recursion used in runner
# Instrumentation writes are intentionally excluded.
# ============================================================

def aasf_exact(
    action,
):
    action = np.asarray(
        action
    )

    smoothed_action = (
        action.copy()
    )

    for t in range(
        1,
        smoothed_action.shape[1],
    ):
        delta = np.linalg.norm(
            action[:, t]
            - smoothed_action[:, t - 1],
            axis=-1,
            keepdims=True,
        )

        excess = np.maximum(
            delta
            - AASF_THRESHOLD,
            0.0,
        ) / (
            AASF_THRESHOLD
            + 1.0e-6
        )

        adaptive_alpha = (
            AASF_ALPHA_MAX
            - AASF_GAIN
            * excess
        )

        adaptive_alpha = np.clip(
            adaptive_alpha,
            AASF_ALPHA_MIN,
            AASF_ALPHA_MAX,
        )

        smoothed_action[:, t] = (
            adaptive_alpha
            * action[:, t]
            + (
                1.0
                - adaptive_alpha
            )
            * smoothed_action[
                :, t - 1
            ]
        )

    return smoothed_action


# ============================================================
# Select a real boundary from seed 4300100
# Prefer a BSOP-active one.
# ============================================================

chosen_query = None

for qidx in range(
    1,
    action_pred.shape[0],
):
    prev_pred_index = (
        action_slice_start
        + K
    )

    y_index = (
        qidx * K
        - 1
    )

    if (
        prev_pred_index
        >= H
        or y_index
        >= len(executed_action)
    ):
        continue

    candidate_y = executed_action[
        y_index:y_index + 1
    ]

    candidate_a = action_pred[
        qidx:qidx + 1,
        action_slice_start,
        :,
    ]

    candidate_p = action_pred[
        qidx - 1:qidx,
        prev_pred_index,
        :,
    ]

    d = (
        candidate_a.astype(
            np.float64
        )
        - candidate_y.astype(
            np.float64
        )
    )

    qq = (
        candidate_p.astype(
            np.float64
        )
        - candidate_a.astype(
            np.float64
        )
    )

    qnorm2 = np.sum(
        qq * qq,
        axis=-1,
    )

    if qnorm2[0] <= 1e-12:
        continue

    lambda_star = (
        -np.sum(
            d * qq,
            axis=-1,
        )
        / qnorm2
    )

    lambda_star = np.clip(
        lambda_star,
        0.0,
        1.0,
    )

    if lambda_star[0] > 0.0:
        chosen_query = qidx
        break


if chosen_query is None:
    chosen_query = 1


Q = chosen_query

Y = executed_action[
    Q * K - 1:
    Q * K
].copy()

A = action_pred[
    Q:Q + 1,
    action_slice_start,
    :,
].copy()

P = action_pred[
    Q - 1:Q,
    action_slice_start + K,
    :,
].copy()

AASF_CHUNK = action_pred[
    Q:Q + 1,
    action_slice_start:
    action_slice_start + K,
    :,
].copy()

TE_PREV_PRED = action_pred[
    Q - 1:Q
].copy()

TE_CURRENT_PRED = action_pred[
    Q:Q + 1
].copy()


# ============================================================
# TE benchmark setup
# ============================================================

TE_CORE_HISTORY = []

append_temporal_ensemble_record(
    history=TE_CORE_HISTORY,
    action_pred=TE_PREV_PRED,
    slice_start=action_slice_start,
    exec_start=np.asarray(
        [0],
        dtype=np.int64,
    ),
    query_index=0,
    max_history=TE_MAX_HISTORY,
)


def te_core_exact():
    # Steady-state query:
    # one immediately preceding record already exists.
    return temporal_ensemble_current_chunk(
        history=TE_CORE_HISTORY,
        current_action_pred=(
            TE_CURRENT_PRED
        ),
        current_slice_start=(
            action_slice_start
        ),
        current_exec_start=np.asarray(
            [K],
            dtype=np.int64,
        ),
        current_chunk_length=K,
        beta=TE_BETA,
    )


class TEFullState:
    def __init__(self):
        self.history = []

        append_temporal_ensemble_record(
            history=self.history,
            action_pred=TE_PREV_PRED,
            slice_start=(
                action_slice_start
            ),
            exec_start=np.asarray(
                [0],
                dtype=np.int64,
            ),
            query_index=0,
            max_history=TE_MAX_HISTORY,
        )

        self.exec_start = K

    def step(self):
        exec_start_arr = np.asarray(
            [self.exec_start],
            dtype=np.int64,
        )

        result = (
            temporal_ensemble_current_chunk(
                history=self.history,
                current_action_pred=(
                    TE_CURRENT_PRED
                ),
                current_slice_start=(
                    action_slice_start
                ),
                current_exec_start=(
                    exec_start_arr
                ),
                current_chunk_length=K,
                beta=TE_BETA,
            )
        )

        query_index = int(
            result[
                "current_query_index"
            ]
        )

        append_temporal_ensemble_record(
            history=self.history,
            action_pred=(
                TE_CURRENT_PRED
            ),
            slice_start=(
                action_slice_start
            ),
            exec_start=(
                exec_start_arr
            ),
            query_index=(
                query_index
            ),
            max_history=(
                TE_MAX_HISTORY
            ),
        )

        self.exec_start += K

        return result


te_full_state = TEFullState()


def te_full_exact():
    return te_full_state.step()


def bsop_aasf_exact(
    y,
    a,
    p,
    action,
):
    action = np.asarray(
        action
    )

    if action.ndim != 3:
        raise RuntimeError(
            "BSOP+AASF action must be 3D"
        )

    smoothed_action = (
        action.copy()
    )

    projected_first = bsop_exact(
        y,
        a,
        p,
    )

    if projected_first.shape != (
        action.shape[0],
        action.shape[2],
    ):
        raise RuntimeError(
            "Unexpected BSOP+AASF first-action shape"
        )

    # Assignment intentionally follows the chunk dtype,
    # matching the runner-style action-array update.
    smoothed_action[
        :,
        0,
        :
    ] = projected_first

    # AASF recursion for positions 1..K-1.
    # The CURRENT input remains the raw policy chunk;
    # only the previous state is recursively filtered.
    for t in range(
        1,
        smoothed_action.shape[1],
    ):
        delta = np.linalg.norm(
            action[:, t]
            - smoothed_action[:, t - 1],
            axis=-1,
            keepdims=True,
        )

        excess = np.maximum(
            delta
            - AASF_THRESHOLD,
            0.0,
        ) / (
            AASF_THRESHOLD
            + 1.0e-6
        )

        adaptive_alpha = (
            AASF_ALPHA_MAX
            - AASF_GAIN
            * excess
        )

        adaptive_alpha = np.clip(
            adaptive_alpha,
            AASF_ALPHA_MIN,
            AASF_ALPHA_MAX,
        )

        smoothed_action[:, t] = (
            adaptive_alpha
            * action[:, t]
            + (
                1.0
                - adaptive_alpha
            )
            * smoothed_action[
                :, t - 1
            ]
        )

    return smoothed_action


def bsop_bench():
    return bsop_exact(
        Y,
        A,
        P,
    )


def aasf_bench():
    return aasf_exact(
        AASF_CHUNK
    )


def bsop_aasf_bench():
    return bsop_aasf_exact(
        Y,
        A,
        P,
        AASF_CHUNK,
    )


def noop_bench():
    return AASF_CHUNK


# ============================================================
# Sanity checks before timing
# ============================================================

bsop_out = bsop_bench()

if bsop_out.shape != (
    1,
    D,
):
    raise RuntimeError(
        "Unexpected BSOP output shape"
    )

aasf_out = aasf_bench()

if aasf_out.shape != (
    1,
    K,
    D,
):
    raise RuntimeError(
        "Unexpected AASF output shape"
    )


# ------------------------------------------------------------
# BSOP+AASF equivalence sanity checks
# ------------------------------------------------------------

bsop_aasf_out = (
    bsop_aasf_bench()
)

# Guard 1: complete output shape.
if bsop_aasf_out.shape != (
    1,
    K,
    D,
):
    raise RuntimeError(
        "Unexpected BSOP+AASF output shape"
    )


# Guard 2: first action must be exactly the independently
# computed BSOP projection after the same chunk-dtype cast.
expected_first = np.asarray(
    bsop_out,
    dtype=bsop_aasf_out.dtype,
)

if not np.array_equal(
    bsop_aasf_out[
        :,
        0,
        :
    ],
    expected_first,
):
    raise RuntimeError(
        "BSOP+AASF first action does not "
        "match independent BSOP output."
    )


# Guard 3: independently reconstruct the AASF recurrence
# from the same projected first action and require the
# remaining seven actions to match exactly.
manual_combined = (
    AASF_CHUNK.copy()
)

manual_combined[
    :,
    0,
    :
] = bsop_out

for t in range(
    1,
    manual_combined.shape[1],
):
    delta = np.linalg.norm(
        AASF_CHUNK[:, t]
        - manual_combined[:, t - 1],
        axis=-1,
        keepdims=True,
    )

    excess = np.maximum(
        delta
        - AASF_THRESHOLD,
        0.0,
    ) / (
        AASF_THRESHOLD
        + 1.0e-6
    )

    adaptive_alpha = (
        AASF_ALPHA_MAX
        - AASF_GAIN
        * excess
    )

    adaptive_alpha = np.clip(
        adaptive_alpha,
        AASF_ALPHA_MIN,
        AASF_ALPHA_MAX,
    )

    manual_combined[:, t] = (
        adaptive_alpha
        * AASF_CHUNK[:, t]
        + (
            1.0
            - adaptive_alpha
        )
        * manual_combined[
            :, t - 1
        ]
    )

if not np.array_equal(
    bsop_aasf_out[
        :,
        1:,
        :
    ],
    manual_combined[
        :,
        1:,
        :
    ],
):
    raise RuntimeError(
        "BSOP+AASF tail does not match "
        "independent AASF recurrence."
    )


te_core_out = te_core_exact()

if not isinstance(
    te_core_out,
    dict,
):
    raise RuntimeError(
        "Unexpected TE output type"
    )

te_full_test = te_full_exact()

if not isinstance(
    te_full_test,
    dict,
):
    raise RuntimeError(
        "Unexpected TE full output type"
    )


# Verify overlap really exists in the TE
# steady-state benchmark.
if "counts" not in te_core_out:
    raise RuntimeError(
        "TE output does not contain expected "
        "'counts' overlap diagnostic."
    )

source_count = np.asarray(
    te_core_out[
        "counts"
    ]
)

if np.max(
    source_count
) < 2:
    raise RuntimeError(
        "TE benchmark did not "
        "exercise temporal overlap."
    )


# ============================================================
# Warm-up
# ============================================================

methods = {
    "noop_python_call": (
        noop_bench
    ),
    "BSOP": (
        bsop_bench
    ),
    "AASF": (
        aasf_bench
    ),
    "BSOP+AASF": (
        bsop_aasf_bench
    ),
    "TE_beta2_core": (
        te_core_exact
    ),
    "TE_beta2_full": (
        te_full_exact
    ),
}


print(
    "===== SANITY ====="
)

print(
    "formal trace:",
    MAIN_PATH,
)

print(
    "overlap trace:",
    DIAG_PATH,
)

print(
    "chosen real query:",
    Q,
)

print(
    "action shape:",
    AASF_CHUNK.shape,
)

print(
    "BSOP y:",
    Y,
)

print(
    "BSOP a:",
    A,
)

print(
    "BSOP p:",
    P,
)

print(
    "BSOP projected:",
    bsop_out,
)

print(
    "BSOP+AASF first:",
    bsop_aasf_out[
        :,
        0,
        :
    ],
)

print(
    "BSOP+AASF equivalence:",
    "PASS",
)

print(
    "TE output keys:",
    sorted(
        te_core_out.keys()
    ),
)

print(
    "TE counts:",
    np.asarray(
        te_core_out[
            "counts"
        ]
    ),
)

print(
    "CPU:",
    machine_info[
        "cpu_model"
    ],
)

print(
    "affinity:",
    machine_info[
        "pin_status"
    ],
)

print(
    "NumPy:",
    np.__version__,
)

print()


print(
    "Warm-up:",
    WARMUP_CALLS,
    "calls per method"
)

for name, fn in methods.items():
    for _ in range(
        WARMUP_CALLS
    ):
        fn()

print(
    "Warm-up complete."
)


# Disable cyclic GC during the measured region.
# Normal Python reference counting remains active.
gc.collect()
gc.disable()


# ============================================================
# Interleaved block timing
# ============================================================

rng = random.Random(
    RNG_SEED
)

method_names = list(
    methods.keys()
)

rows = []


print()
print(
    "===== MEASURED BENCHMARK ====="
)

print(
    f"{N_BLOCKS} blocks x "
    f"{CALLS_PER_BLOCK} calls"
)

overall_start = time.perf_counter()


for block_idx in range(
    N_BLOCKS
):
    order = method_names.copy()

    rng.shuffle(
        order
    )

    for order_idx, name in enumerate(
        order
    ):
        fn = methods[
            name
        ]

        t0 = time.perf_counter_ns()

        for _ in range(
            CALLS_PER_BLOCK
        ):
            fn()

        t1 = time.perf_counter_ns()

        elapsed_ns = (
            t1
            - t0
        )

        us_per_call = (
            elapsed_ns
            / CALLS_PER_BLOCK
            / 1000.0
        )

        rows.append({
            "block": block_idx,
            "order_in_block": (
                order_idx
            ),
            "method": name,
            "calls": (
                CALLS_PER_BLOCK
            ),
            "elapsed_ns": (
                elapsed_ns
            ),
            "us_per_query": (
                us_per_call
            ),
        })

    if (
        (block_idx + 1) % 20
        == 0
    ):
        print(
            "completed",
            block_idx + 1,
            "/",
            N_BLOCKS,
        )


overall_end = time.perf_counter()

gc.enable()


# ============================================================
# Save raw block timing
# ============================================================

BLOCK_CSV = (
    OUT_DIR
    / "timing_blocks.csv"
)

with BLOCK_CSV.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "block",
            "order_in_block",
            "method",
            "calls",
            "elapsed_ns",
            "us_per_query",
        ],
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# ============================================================
# Summaries
# ============================================================

summary_rows = []

for name in method_names:
    values = np.asarray(
        [
            row[
                "us_per_query"
            ]
            for row in rows
            if row[
                "method"
            ] == name
        ],
        dtype=np.float64,
    )

    summary_rows.append({
        "method": name,
        "n_blocks": int(
            len(values)
        ),
        "calls_per_block": (
            CALLS_PER_BLOCK
        ),
        "total_calls": int(
            len(values)
            * CALLS_PER_BLOCK
        ),
        "mean_us_per_query": float(
            np.mean(values)
        ),
        "sd_us_per_query": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "median_us_per_query": float(
            np.median(values)
        ),
        "p95_us_per_query": float(
            np.percentile(
                values,
                95,
            )
        ),
        "min_us_per_query": float(
            np.min(values)
        ),
        "max_us_per_query": float(
            np.max(values)
        ),
    })


SUMMARY_CSV = (
    OUT_DIR
    / "timing_summary.csv"
)

with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


machine_info[
    "elapsed_benchmark_seconds"
] = (
    overall_end
    - overall_start
)

machine_info[
    "chosen_formal_query"
] = Q

machine_info[
    "main_trace"
] = str(
    MAIN_PATH
)

machine_info[
    "diag_trace"
] = str(
    DIAG_PATH
)


MACHINE_JSON = (
    OUT_DIR
    / "machine_and_protocol.json"
)

with MACHINE_JSON.open(
    "w"
) as f:
    json.dump(
        machine_info,
        f,
        indent=2,
    )


# SHA256 of this benchmark script.
try:
    script_path = Path(
        __file__
    )

    digest = hashlib.sha256(
        script_path.read_bytes()
    ).hexdigest()

    (
        OUT_DIR
        / "script_sha256.txt"
    ).write_text(
        digest
        + "  "
        + script_path.name
        + "\n"
    )
except Exception:
    pass


# Human-readable output
print()
print(
    "===== FINAL SUMMARY ====="
)

header = (
    f"{'Method':<20}"
    f"{'Mean us':>12}"
    f"{'SD us':>12}"
    f"{'Median us':>12}"
    f"{'P95 us':>12}"
)

print(
    header
)

print(
    "-" * len(header)
)

for row in summary_rows:
    print(
        f"{row['method']:<20}"
        f"{row['mean_us_per_query']:>12.4f}"
        f"{row['sd_us_per_query']:>12.4f}"
        f"{row['median_us_per_query']:>12.4f}"
        f"{row['p95_us_per_query']:>12.4f}"
    )


print()
print(
    "Measured region elapsed:",
    f"{machine_info['elapsed_benchmark_seconds']:.2f}",
    "s",
)

print()
print(
    "Saved:"
)

print(
    BLOCK_CSV
)

print(
    SUMMARY_CSV
)

print(
    MACHINE_JSON
)

print(
    OUT_DIR
    / "script_sha256.txt"
)

print()
print(
    "IMPORTANT: P95 is the 95th percentile "
    "of block-averaged per-query latency."
)

print(
    "Do not subtract the no-op timing; "
    "it is reported only as a timing floor."
)
