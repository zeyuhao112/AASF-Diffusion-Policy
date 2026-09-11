from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# Frozen SG development study
# ============================================================

ROOT = Path(
    "paper_exp/savgol_baseline_seed42"
)

GRID_ROOT = ROOT / "development_grid"
TRACE_ROOT = GRID_ROOT / "traces"
RESULT_ROOT = GRID_ROOT / "results"

ANALYSIS_ROOT = ROOT / "development_analysis"
SELECTION_ROOT = ROOT / "selection"

PROTOCOL = ROOT / "SG_PROTOCOL_FREEZE.txt"
FREEZE = ROOT / "SG_PROTOCOL_FREEZE_SHA256.txt"

GRID_MANIFEST = (
    GRID_ROOT / "SG_DEVELOPMENT_GRID_SHA256.txt"
)

RUNNER = Path(
    "diffusion_policy/env_runner/"
    "pusht_keypoints_runner.py"
)

CHECKPOINT = Path(
    "data/checkpoints/pusht.ckpt"
)

DEV_SEEDS = list(
    range(4300050, 4300070)
)

FORMAL_CONFIRMATORY_SEEDS = (
    4300100,
    4300149,
)

METHODS = [
    ("original", None, None),
    ("w3_p1", 3, 1),
    ("w3_p2", 3, 2),
    ("w5_p1", 5, 1),
    ("w5_p2", 5, 2),
    ("w5_p3", 5, 3),
    ("w7_p1", 7, 1),
    ("w7_p2", 7, 2),
    ("w7_p3", 7, 3),
]

SG_METHODS = METHODS[1:]

SR90_THRESHOLD = 0.90
MEAN_SCORE_DIFF_FLOOR = -0.01
MAX_SR90_COUNT_DECREASE = 1

SELECTION_DECIMALS = 6
CONTROL_DT = 0.1

BOOTSTRAP_SAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260816


EXPECTED_SHA256 = {
    RUNNER:
        "02e6a88caf26cf3a7509034f95a1e896a"
        "782274d37b1d9965253ac6ebf4a245d",

    PROTOCOL:
        "b0ba2312b017c4f62ad4216d6a8a9898"
        "9eeb1eaba8521dcd4531317aaadb7b10",

    GRID_MANIFEST:
        "063a79567e7a29488381509cdd55071ff"
        "790cfd7947b6263b251556f66589b9f",

    CHECKPOINT:
        "f804e16575e261fa0b7e981da3f67741"
        "fc8517817734320d550e43a4182bf876",
}


def sha256_of(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def verify_fixed_sha256():
    for path, expected in EXPECTED_SHA256.items():

        if not path.is_file():
            raise RuntimeError(
                f"Frozen input missing: {path}"
            )

        actual = sha256_of(path)

        if actual != expected:
            raise RuntimeError(
                "Frozen input SHA256 mismatch:\n"
                f"path={path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )


# ============================================================
# Canonical smoothness metrics
# ============================================================

def finite_difference_metric(
    trajectory,
    order,
    dt=1.0,
):
    """
    Mean L2 norm of the order-th finite difference:

        mean_t ||Delta^order u_t||_2 / dt^order

    dt=1 gives the raw command-space D1/D2/D3 metrics.
    dt=CONTROL_DT gives time-normalized C1/C2/C3.
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    order = int(order)
    dt = float(dt)

    if trajectory.ndim != 2:
        raise RuntimeError(
            "Expected trajectory shape "
            "(time, action_dim), got "
            f"{trajectory.shape}"
        )

    if order < 1:
        raise RuntimeError(
            f"Finite-difference order must be >= 1; "
            f"got {order}"
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError(
            f"dt must be finite and strictly positive; "
            f"got {dt}"
        )

    if trajectory.shape[0] <= order:
        raise RuntimeError(
            "Trajectory is too short for "
            f"order={order}: "
            f"length={trajectory.shape[0]}"
        )

    d = trajectory

    for _ in range(order):
        d = np.diff(
            d,
            axis=0,
        )

    scale = dt ** order

    return float(
        np.mean(
            np.linalg.norm(
                d,
                axis=-1,
            )
        )
        / scale
    )


def boundary_d1_metric(
    executed_action,
    executed_steps_per_query,
):
    """
    Own-trajectory Boundary D1.

    A boundary occurs between the last actually executed
    control of one policy query and the first actually
    executed control of the next query.

    Planned chunk length is intentionally not used.
    """

    action = np.asarray(
        executed_action,
        dtype=np.float64,
    )

    counts = np.asarray(
        executed_steps_per_query,
        dtype=np.int64,
    )

    if action.ndim != 2:
        raise RuntimeError(
            "executed_action must be 2D; "
            f"got {action.shape}"
        )

    if counts.ndim != 1:
        raise RuntimeError(
            "executed_steps_per_query must be 1D"
        )

    if len(counts) < 2:
        raise RuntimeError(
            "At least two actually executed policy "
            "queries are required for Boundary D1."
        )

    if np.any(counts <= 0):
        raise RuntimeError(
            "Encountered a policy query with "
            "non-positive actual execution count."
        )

    total = int(
        np.sum(
            counts,
            dtype=np.int64,
        )
    )

    if total != int(action.shape[0]):
        raise RuntimeError(
            "Actual execution accounting mismatch: "
            f"sum(counts)={total}, "
            f"trajectory_length={action.shape[0]}"
        )

    starts = np.cumsum(
        counts,
        dtype=np.int64,
    )[:-1]

    if np.any(starts <= 0):
        raise RuntimeError(
            "Invalid policy-query boundary index."
        )

    if np.any(starts >= action.shape[0]):
        raise RuntimeError(
            "Policy-query boundary lies outside "
            "the actually executed trajectory."
        )

    jumps = (
        action[starts]
        - action[starts - 1]
    )

    value = float(
        np.mean(
            np.linalg.norm(
                jumps,
                axis=-1,
            )
        )
    )

    return value


def percent_reduction(
    old,
    new,
):
    old = float(old)
    new = float(new)

    if not np.isfinite(old):
        raise RuntimeError(
            "Non-finite baseline metric."
        )

    if not np.isfinite(new):
        raise RuntimeError(
            "Non-finite comparison metric."
        )

    if old <= 0.0:
        raise RuntimeError(
            "Percentage reduction requires a "
            "strictly positive Original metric; "
            f"got {old}"
        )

    return float(
        100.0
        * (old - new)
        / old
    )


def paired_common_prefix_metrics(
    original_action,
    sg_action,
):
    """
    Compute D1/D2/D3 and C1/C2/C3 on the prespecified
    paired common actually executed prefix.

    This function does NOT compute Boundary D1, because
    Boundary D1 is intentionally an own-trajectory metric.
    """

    original_action = np.asarray(
        original_action,
        dtype=np.float64,
    )

    sg_action = np.asarray(
        sg_action,
        dtype=np.float64,
    )

    if original_action.ndim != 2:
        raise RuntimeError(
            "Original executed action must be 2D; "
            f"got {original_action.shape}"
        )

    if sg_action.ndim != 2:
        raise RuntimeError(
            "SG executed action must be 2D; "
            f"got {sg_action.shape}"
        )

    if (
        original_action.shape[1]
        != sg_action.shape[1]
    ):
        raise RuntimeError(
            "Action dimensions differ between pair: "
            f"original={original_action.shape}, "
            f"sg={sg_action.shape}"
        )

    common_prefix_steps = min(
        int(original_action.shape[0]),
        int(sg_action.shape[0]),
    )

    if common_prefix_steps < 4:
        raise RuntimeError(
            "Paired common executed prefix is too short "
            "for D3: "
            f"{common_prefix_steps}"
        )

    original_prefix = (
        original_action[:common_prefix_steps]
    )

    sg_prefix = (
        sg_action[:common_prefix_steps]
    )

    result = {
        "common_prefix_steps":
            common_prefix_steps,
    }

    for order in (1, 2, 3):
        d_old = finite_difference_metric(
            original_prefix,
            order,
            dt=1.0,
        )

        d_new = finite_difference_metric(
            sg_prefix,
            order,
            dt=1.0,
        )

        c_old = finite_difference_metric(
            original_prefix,
            order,
            dt=CONTROL_DT,
        )

        c_new = finite_difference_metric(
            sg_prefix,
            order,
            dt=CONTROL_DT,
        )

        d_reduction = percent_reduction(
            d_old,
            d_new,
        )

        c_reduction = percent_reduction(
            c_old,
            c_new,
        )

        if not np.isclose(
            d_reduction,
            c_reduction,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                "Time-scaling reduction invariance "
                f"failed for order={order}: "
                f"D={d_reduction}, "
                f"C={c_reduction}"
            )

        result[f"original_D{order}"] = d_old
        result[f"sg_D{order}"] = d_new

        result[
            f"D{order}_reduction_pct"
        ] = d_reduction

        result[
            f"original_C{order}"
        ] = c_old

        result[
            f"sg_C{order}"
        ] = c_new

        result[
            f"C{order}_reduction_pct"
        ] = c_reduction

    return result


# ============================================================
# Development trace loading and execution-accounting audit
# ============================================================

TRACE_METHODS = (
    "original",
    "w3_p1",
    "w3_p2",
    "w5_p1",
    "w5_p2",
    "w5_p3",
    "w7_p1",
    "w7_p2",
    "w7_p3",
)


def normal_trace_path(
    method,
    seed,
):
    method = str(method)
    seed = int(seed)

    if method not in TRACE_METHODS:
        raise RuntimeError(
            f"Unknown development method: {method}"
        )

    return (
        TRACE_ROOT
        / method
        / f"test_seed_{seed}.npz"
    )


def audit_method_trace_set(
    method,
):
    """
    Require exactly one normal trace for every frozen
    development seed and no unexpected normal trace.
    """

    method = str(method)

    if method not in TRACE_METHODS:
        raise RuntimeError(
            f"Unknown development method: {method}"
        )

    method_dir = (
        TRACE_ROOT
        / method
    )

    if not method_dir.is_dir():
        raise RuntimeError(
            "Development trace directory missing: "
            f"{method_dir}"
        )

    expected_names = {
        f"test_seed_{seed}.npz"
        for seed in DEV_SEEDS
    }

    actual_names = {
        p.name
        for p in method_dir.glob(
            "test_seed_*.npz"
        )
        if not p.name.endswith(
            "_overlap_diag.npz"
        )
    }

    missing = sorted(
        expected_names
        - actual_names
    )

    unexpected = sorted(
        actual_names
        - expected_names
    )

    if missing:
        raise RuntimeError(
            f"{method}: missing normal traces: "
            f"{missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"{method}: unexpected normal traces: "
            f"{unexpected}"
        )

    if len(actual_names) != len(DEV_SEEDS):
        raise RuntimeError(
            f"{method}: expected "
            f"{len(DEV_SEEDS)} normal traces, "
            f"found {len(actual_names)}"
        )

    return len(actual_names)


def load_trace_record(
    path,
    expected_seed,
):
    """
    Load one normal execution trace and mechanically
    verify actual-execution accounting.

    No task-performance or smoothing comparison is made
    here. This function only establishes that the trace
    is internally consistent and returns the frozen
    quantities needed by later paired analysis.
    """

    path = Path(path)
    expected_seed = int(
        expected_seed
    )

    if not path.is_file():
        raise RuntimeError(
            f"Trace file missing: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        required = {
            "seed",
            "score",
            "executed_action",
            "executed_steps_per_query",
            "cumulative_executed_steps",
            "actual_executed_steps",
            "planned_steps_per_query",
            "planned_action_steps_total",
            "unexecuted_steps_per_query",
            "unexecuted_action_steps_total",
            "final_chunk_executed_steps",
            "final_chunk_unexecuted_steps",
        }

        missing = (
            required
            - set(z.files)
        )

        if missing:
            raise RuntimeError(
                f"{path}: missing required keys "
                f"{sorted(missing)}"
            )

        seed = int(
            np.asarray(
                z["seed"]
            ).item()
        )

        score = float(
            np.asarray(
                z["score"]
            ).item()
        )

        action = np.asarray(
            z["executed_action"],
            dtype=np.float64,
        )

        counts = np.asarray(
            z[
                "executed_steps_per_query"
            ],
            dtype=np.int64,
        )

        cumulative = np.asarray(
            z[
                "cumulative_executed_steps"
            ],
            dtype=np.int64,
        )

        actual_executed_steps = int(
            np.asarray(
                z[
                    "actual_executed_steps"
                ]
            ).item()
        )

        planned = np.asarray(
            z[
                "planned_steps_per_query"
            ],
            dtype=np.int64,
        )

        planned_total = int(
            np.asarray(
                z[
                    "planned_action_steps_total"
                ]
            ).item()
        )

        unexecuted = np.asarray(
            z[
                "unexecuted_steps_per_query"
            ],
            dtype=np.int64,
        )

        unexecuted_total = int(
            np.asarray(
                z[
                    "unexecuted_action_steps_total"
                ]
            ).item()
        )

        final_executed = int(
            np.asarray(
                z[
                    "final_chunk_executed_steps"
                ]
            ).item()
        )

        final_unexecuted = int(
            np.asarray(
                z[
                    "final_chunk_unexecuted_steps"
                ]
            ).item()
        )

        if "chunk_starts" in z.files:
            chunk_starts = np.asarray(
                z["chunk_starts"],
                dtype=np.int64,
            )
        else:
            chunk_starts = None

    if seed != expected_seed:
        raise RuntimeError(
            f"{path}: seed mismatch: "
            f"expected={expected_seed}, "
            f"payload={seed}"
        )

    if not np.isfinite(score):
        raise RuntimeError(
            f"{path}: non-finite score"
        )

    if action.ndim != 2:
        raise RuntimeError(
            f"{path}: executed_action must "
            f"be 2D, got {action.shape}"
        )

    if not np.all(
        np.isfinite(action)
    ):
        raise RuntimeError(
            f"{path}: executed_action contains "
            "non-finite values"
        )

    for name, array in (
        (
            "executed_steps_per_query",
            counts,
        ),
        (
            "cumulative_executed_steps",
            cumulative,
        ),
        (
            "planned_steps_per_query",
            planned,
        ),
        (
            "unexecuted_steps_per_query",
            unexecuted,
        ),
    ):
        if array.ndim != 1:
            raise RuntimeError(
                f"{path}: {name} must be 1D"
            )

    n_queries = int(
        len(counts)
    )

    if n_queries == 0:
        raise RuntimeError(
            f"{path}: no policy queries"
        )

    if len(cumulative) != n_queries:
        raise RuntimeError(
            f"{path}: cumulative/count "
            "length mismatch"
        )

    if len(planned) != n_queries:
        raise RuntimeError(
            f"{path}: planned/count "
            "length mismatch"
        )

    if len(unexecuted) != n_queries:
        raise RuntimeError(
            f"{path}: unexecuted/count "
            "length mismatch"
        )

    if np.any(counts <= 0):
        raise RuntimeError(
            f"{path}: non-positive actual "
            "executed query length"
        )

    if np.any(planned <= 0):
        raise RuntimeError(
            f"{path}: non-positive planned "
            "query length"
        )

    if np.any(unexecuted < 0):
        raise RuntimeError(
            f"{path}: negative unexecuted "
            "query length"
        )

    if np.any(
        counts + unexecuted
        != planned
    ):
        raise RuntimeError(
            f"{path}: per-query accounting "
            "does not satisfy "
            "executed + unexecuted = planned"
        )

    expected_cumulative = np.cumsum(
        counts,
        dtype=np.int64,
    )

    if not np.array_equal(
        cumulative,
        expected_cumulative,
    ):
        raise RuntimeError(
            f"{path}: cumulative actual "
            "execution accounting mismatch"
        )

    count_total = int(
        np.sum(
            counts,
            dtype=np.int64,
        )
    )

    planned_sum = int(
        np.sum(
            planned,
            dtype=np.int64,
        )
    )

    unexecuted_sum = int(
        np.sum(
            unexecuted,
            dtype=np.int64,
        )
    )

    if count_total != int(
        action.shape[0]
    ):
        raise RuntimeError(
            f"{path}: executed_action length "
            "does not equal sum of actual "
            "execution counts: "
            f"action={action.shape[0]}, "
            f"counts={count_total}"
        )

    if actual_executed_steps != count_total:
        raise RuntimeError(
            f"{path}: actual_executed_steps "
            "mismatch"
        )

    if int(cumulative[-1]) != count_total:
        raise RuntimeError(
            f"{path}: final cumulative count "
            "mismatch"
        )

    if planned_total != planned_sum:
        raise RuntimeError(
            f"{path}: planned total mismatch"
        )

    if unexecuted_total != unexecuted_sum:
        raise RuntimeError(
            f"{path}: unexecuted total mismatch"
        )

    if (
        actual_executed_steps
        + unexecuted_total
        != planned_total
    ):
        raise RuntimeError(
            f"{path}: total accounting does "
            "not satisfy actual + unexecuted "
            "= planned"
        )

    if final_executed != int(
        counts[-1]
    ):
        raise RuntimeError(
            f"{path}: final executed chunk "
            "count mismatch"
        )

    if final_unexecuted != int(
        unexecuted[-1]
    ):
        raise RuntimeError(
            f"{path}: final unexecuted chunk "
            "count mismatch"
        )

    if chunk_starts is not None:

        if chunk_starts.ndim != 1:
            raise RuntimeError(
                f"{path}: chunk_starts "
                "must be 1D"
            )

        if len(chunk_starts) != n_queries:
            raise RuntimeError(
                f"{path}: chunk_starts "
                "length mismatch"
            )

        expected_starts = np.concatenate(
            (
                np.asarray(
                    [0],
                    dtype=np.int64,
                ),
                cumulative[:-1],
            )
        )

        if not np.array_equal(
            chunk_starts,
            expected_starts,
        ):
            raise RuntimeError(
                f"{path}: chunk_starts "
                "do not match actual "
                "execution boundaries"
            )

    boundary_d1 = (
        boundary_d1_metric(
            action,
            counts,
        )
    )

    return {
        "path":
            str(path),

        "seed":
            seed,

        "score":
            score,

        "sr90":
            int(
                score
                >= SR90_THRESHOLD
            ),

        "executed_action":
            action,

        "executed_steps_per_query":
            counts,

        "actual_executed_steps":
            actual_executed_steps,

        "n_queries":
            n_queries,

        "boundary_D1":
            boundary_d1,
    }


def load_development_method(
    method,
):
    """
    Load all 20 frozen development traces for one method.
    """

    audit_method_trace_set(
        method
    )

    records = {}

    for seed in DEV_SEEDS:
        path = normal_trace_path(
            method,
            seed,
        )

        records[seed] = (
            load_trace_record(
                path,
                seed,
            )
        )

    if set(records) != set(
        DEV_SEEDS
    ):
        raise RuntimeError(
            f"{method}: loaded seed set "
            "does not equal frozen "
            "development seed set"
        )

    return records


def audit_all_development_traces():
    """
    Structural audit only.

    Loads every frozen development normal trace and checks
    exact seed identity plus actual/planned/unexecuted
    execution accounting.

    No candidate ranking or selection occurs here.
    """

    audited = {}

    for method in TRACE_METHODS:
        records = (
            load_development_method(
                method
            )
        )

        audited[method] = int(
            len(records)
        )

    total = int(
        sum(
            audited.values()
        )
    )

    expected_total = (
        len(TRACE_METHODS)
        * len(DEV_SEEDS)
    )

    if total != expected_total:
        raise RuntimeError(
            "Development trace accounting "
            "total mismatch: "
            f"expected={expected_total}, "
            f"actual={total}"
        )

    return {
        "per_method":
            audited,

        "total_normal_traces":
            total,
    }


# ============================================================
# Frozen SG paired development metrics
# ============================================================

SG_CANDIDATE_METHODS = (
    "w3_p1",
    "w3_p2",
    "w5_p1",
    "w5_p2",
    "w5_p3",
    "w7_p1",
    "w7_p2",
    "w7_p3",
)

SG_CONTROL_DT = 0.1


def sg_time_scaled_difference_metric(
    trajectory,
    order,
    dt,
):
    """
    Mean L2 norm of the requested finite difference,
    normalized by physical control time:

        mean_t || Delta^order u_t / dt^order ||_2

    This helper is intentionally local to the frozen SG
    development analysis.  Percentage reductions must be
    invariant to the constant dt scaling.
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    order = int(order)
    dt = float(dt)

    if trajectory.ndim != 2:
        raise RuntimeError(
            "SG finite-difference input must be "
            "2D (time, action_dim), got "
            f"{trajectory.shape}"
        )

    if order < 1:
        raise RuntimeError(
            f"SG finite-difference order must be >= 1, "
            f"got {order}"
        )

    if trajectory.shape[0] <= order:
        raise RuntimeError(
            "SG trajectory too short for "
            f"order={order}: "
            f"length={trajectory.shape[0]}"
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError(
            f"Invalid SG control dt: {dt}"
        )

    if not np.all(
        np.isfinite(trajectory)
    ):
        raise RuntimeError(
            "SG finite-difference input contains "
            "non-finite values"
        )

    d = trajectory

    for _ in range(order):
        d = np.diff(
            d,
            axis=0,
        )

    d = (
        d
        / (dt ** order)
    )

    return float(
        np.mean(
            np.linalg.norm(
                d,
                axis=-1,
            )
        )
    )


def sg_common_prefix_metrics(
    original_action,
    sg_action,
):
    """
    Frozen paired common-prefix smoothness comparison.

    D1/D2/D3 use dt=1 command-space finite differences.
    C1/C2/C3 use the same common prefix with dt=0.1 s.

    The common prefix is exactly:

        min(
            original actual executed length,
            SG actual executed length,
        )

    Own-trajectory Boundary-D1 is deliberately NOT
    computed here.
    """

    original_action = np.asarray(
        original_action,
        dtype=np.float64,
    )

    sg_action = np.asarray(
        sg_action,
        dtype=np.float64,
    )

    if original_action.ndim != 2:
        raise RuntimeError(
            "Original executed action must be 2D"
        )

    if sg_action.ndim != 2:
        raise RuntimeError(
            "SG executed action must be 2D"
        )

    if (
        original_action.shape[1]
        != sg_action.shape[1]
    ):
        raise RuntimeError(
            "Original/SG action dimensions differ: "
            f"original={original_action.shape}, "
            f"sg={sg_action.shape}"
        )

    common_prefix_steps = min(
        int(original_action.shape[0]),
        int(sg_action.shape[0]),
    )

    if common_prefix_steps < 4:
        raise RuntimeError(
            "SG paired common prefix is too short "
            "for D3: "
            f"{common_prefix_steps}"
        )

    original_prefix = (
        original_action[
            :common_prefix_steps
        ]
    )

    sg_prefix = (
        sg_action[
            :common_prefix_steps
        ]
    )

    result = {
        "common_prefix_steps":
            common_prefix_steps,
    }

    for order in (
        1,
        2,
        3,
    ):

        original_d = (
            sg_time_scaled_difference_metric(
                original_prefix,
                order,
                1.0,
            )
        )

        sg_d = (
            sg_time_scaled_difference_metric(
                sg_prefix,
                order,
                1.0,
            )
        )

        original_c = (
            sg_time_scaled_difference_metric(
                original_prefix,
                order,
                SG_CONTROL_DT,
            )
        )

        sg_c = (
            sg_time_scaled_difference_metric(
                sg_prefix,
                order,
                SG_CONTROL_DT,
            )
        )

        d_reduction = percent_reduction(
            original_d,
            sg_d,
        )

        c_reduction = percent_reduction(
            original_c,
            sg_c,
        )

        if not np.isclose(
            d_reduction,
            c_reduction,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                "SG time-scaling reduction "
                f"invariance failed for order={order}: "
                f"D{order}={d_reduction}, "
                f"C{order}={c_reduction}"
            )

        result[
            f"original_D{order}"
        ] = original_d

        result[
            f"sg_D{order}"
        ] = sg_d

        result[
            f"D{order}_reduction_pct"
        ] = d_reduction

        result[
            f"original_C{order}"
        ] = original_c

        result[
            f"sg_C{order}"
        ] = sg_c

        result[
            f"C{order}_reduction_pct"
        ] = c_reduction

    return result


def build_sg_paired_development_rows():
    """
    Construct the complete frozen 8 x 20 SG development
    paired table in memory.

    No eligibility filtering, ranking, tie breaking,
    selection artifact, or confirmatory access occurs.
    """

    original_records = (
        load_development_method(
            "original"
        )
    )

    rows = []

    for method in SG_CANDIDATE_METHODS:

        sg_records = (
            load_development_method(
                method
            )
        )

        for seed in DEV_SEEDS:

            original = (
                original_records[
                    seed
                ]
            )

            sg = (
                sg_records[
                    seed
                ]
            )

            if (
                int(original["seed"])
                != int(seed)
            ):
                raise RuntimeError(
                    "Original seed identity "
                    f"failure for seed={seed}"
                )

            if (
                int(sg["seed"])
                != int(seed)
            ):
                raise RuntimeError(
                    "SG seed identity "
                    f"failure for method={method}, "
                    f"seed={seed}"
                )

            common = (
                sg_common_prefix_metrics(
                    original[
                        "executed_action"
                    ],
                    sg[
                        "executed_action"
                    ],
                )
            )

            boundary_reduction = (
                percent_reduction(
                    original[
                        "boundary_D1"
                    ],
                    sg[
                        "boundary_D1"
                    ],
                )
            )

            row = {
                "method":
                    method,

                "seed":
                    int(seed),

                "original_score":
                    float(
                        original[
                            "score"
                        ]
                    ),

                "sg_score":
                    float(
                        sg[
                            "score"
                        ]
                    ),

                "score_diff_sg_minus_original":
                    float(
                        sg[
                            "score"
                        ]
                        - original[
                            "score"
                        ]
                    ),

                "original_sr90":
                    int(
                        original[
                            "sr90"
                        ]
                    ),

                "sg_sr90":
                    int(
                        sg[
                            "sr90"
                        ]
                    ),

                "sr90_diff":
                    int(
                        sg[
                            "sr90"
                        ]
                        - original[
                            "sr90"
                        ]
                    ),

                "original_actual_executed_steps":
                    int(
                        original[
                            "actual_executed_steps"
                        ]
                    ),

                "sg_actual_executed_steps":
                    int(
                        sg[
                            "actual_executed_steps"
                        ]
                    ),

                "original_n_queries":
                    int(
                        original[
                            "n_queries"
                        ]
                    ),

                "sg_n_queries":
                    int(
                        sg[
                            "n_queries"
                        ]
                    ),

                "original_boundary_D1":
                    float(
                        original[
                            "boundary_D1"
                        ]
                    ),

                "sg_boundary_D1":
                    float(
                        sg[
                            "boundary_D1"
                        ]
                    ),

                "boundary_D1_reduction_pct":
                    float(
                        boundary_reduction
                    ),
            }

            row.update(
                common
            )

            rows.append(
                row
            )

    expected_pairs = {
        (
            method,
            seed,
        )
        for method
        in SG_CANDIDATE_METHODS
        for seed
        in DEV_SEEDS
    }

    actual_pairs = {
        (
            str(row["method"]),
            int(row["seed"]),
        )
        for row in rows
    }

    if actual_pairs != expected_pairs:
        missing = sorted(
            expected_pairs
            - actual_pairs
        )

        unexpected = sorted(
            actual_pairs
            - expected_pairs
        )

        raise RuntimeError(
            "SG development pair identity "
            "mismatch: "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    expected_count = (
        len(SG_CANDIDATE_METHODS)
        * len(DEV_SEEDS)
    )

    if len(rows) != expected_count:
        raise RuntimeError(
            "SG development paired-row count "
            "mismatch: "
            f"expected={expected_count}, "
            f"actual={len(rows)}"
        )

    return rows


def audit_sg_paired_development_rows(
    rows=None,
):
    """
    Mechanical audit of the in-memory paired development
    records.  Does not rank or select candidates.
    """

    if rows is None:
        rows = (
            build_sg_paired_development_rows()
        )

    expected_count = (
        len(SG_CANDIDATE_METHODS)
        * len(DEV_SEEDS)
    )

    if len(rows) != expected_count:
        raise RuntimeError(
            "Unexpected SG paired-row count"
        )

    per_method = {}

    max_time_scaling_error = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }

    min_common_prefix = None
    max_common_prefix = None

    numeric_fields = (
        "original_score",
        "sg_score",
        "score_diff_sg_minus_original",
        "original_boundary_D1",
        "sg_boundary_D1",
        "boundary_D1_reduction_pct",
        "original_D1",
        "sg_D1",
        "D1_reduction_pct",
        "original_D2",
        "sg_D2",
        "D2_reduction_pct",
        "original_D3",
        "sg_D3",
        "D3_reduction_pct",
        "original_C1",
        "sg_C1",
        "C1_reduction_pct",
        "original_C2",
        "sg_C2",
        "C2_reduction_pct",
        "original_C3",
        "sg_C3",
        "C3_reduction_pct",
    )

    for row in rows:

        method = str(
            row["method"]
        )

        seed = int(
            row["seed"]
        )

        if method not in SG_CANDIDATE_METHODS:
            raise RuntimeError(
                f"Unexpected SG method: {method}"
            )

        if seed not in DEV_SEEDS:
            raise RuntimeError(
                "Non-development seed entered "
                f"SG paired table: {seed}"
            )

        per_method[
            method
        ] = (
            per_method.get(
                method,
                0,
            )
            + 1
        )

        values = np.asarray(
            [
                float(
                    row[field]
                )
                for field
                in numeric_fields
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                "Non-finite SG paired metric: "
                f"method={method}, "
                f"seed={seed}"
            )

        common_prefix_steps = int(
            row[
                "common_prefix_steps"
            ]
        )

        if common_prefix_steps < 4:
            raise RuntimeError(
                "Common prefix below D3 "
                "minimum length"
            )

        if min_common_prefix is None:
            min_common_prefix = (
                common_prefix_steps
            )
            max_common_prefix = (
                common_prefix_steps
            )
        else:
            min_common_prefix = min(
                min_common_prefix,
                common_prefix_steps,
            )
            max_common_prefix = max(
                max_common_prefix,
                common_prefix_steps,
            )

        for order in (
            1,
            2,
            3,
        ):
            error = abs(
                float(
                    row[
                        f"D{order}_reduction_pct"
                    ]
                )
                - float(
                    row[
                        f"C{order}_reduction_pct"
                    ]
                )
            )

            max_time_scaling_error[
                order
            ] = max(
                max_time_scaling_error[
                    order
                ],
                error,
            )

    for method in SG_CANDIDATE_METHODS:
        if per_method.get(
            method,
            0,
        ) != len(DEV_SEEDS):
            raise RuntimeError(
                f"{method}: expected "
                f"{len(DEV_SEEDS)} paired rows, "
                f"found "
                f"{per_method.get(method, 0)}"
            )

    for order in (
        1,
        2,
        3,
    ):
        if (
            max_time_scaling_error[
                order
            ]
            > 1.0e-10
        ):
            raise RuntimeError(
                "SG maximum time-scaling "
                "invariance error exceeded "
                "tolerance for "
                f"order={order}: "
                f"{max_time_scaling_error[order]}"
            )

    return {
        "paired_row_count":
            int(len(rows)),

        "per_method":
            per_method,

        "min_common_prefix_steps":
            int(min_common_prefix),

        "max_common_prefix_steps":
            int(max_common_prefix),

        "max_C1_vs_D1_reduction_error":
            float(
                max_time_scaling_error[
                    1
                ]
            ),

        "max_C2_vs_D2_reduction_error":
            float(
                max_time_scaling_error[
                    2
                ]
            ),

        "max_C3_vs_D3_reduction_error":
            float(
                max_time_scaling_error[
                    3
                ]
            ),
    }


# ============================================================
# Frozen SG development summary and eligibility
# ============================================================

import tempfile


SG_MEAN_SCORE_DIFF_FLOOR = -0.01
SG_MAX_SR90_COUNT_DECREASE = 1

SG_DEVELOPMENT_ANALYSIS_ROOT = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_analysis"
)

SG_ANALYZER_PATH = Path(
    "paper_exp/scripts/"
    "analyze_savgol_dev_seed42.py"
)

SG_RUNNER_PATH = Path(
    "diffusion_policy/env_runner/"
    "pusht_keypoints_runner.py"
)

SG_PROTOCOL_PATH = Path(
    "paper_exp/savgol_baseline_seed42/"
    "SG_PROTOCOL_FREEZE.txt"
)

SG_GRID_MANIFEST_PATH = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_grid/"
    "SG_DEVELOPMENT_GRID_SHA256.txt"
)

SG_CHECKPOINT_PATH = Path(
    "data/checkpoints/pusht.ckpt"
)


SG_METHOD_PARAMETERS = {
    "w3_p1": (3, 1),
    "w3_p2": (3, 2),
    "w5_p1": (5, 1),
    "w5_p2": (5, 2),
    "w5_p3": (5, 3),
    "w7_p1": (7, 1),
    "w7_p2": (7, 2),
    "w7_p3": (7, 3),
}


def sg_sha256_of(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def build_sg_development_summary(
    paired_rows=None,
):
    """
    Build the frozen descriptive development summary and
    apply ONLY the preregistered eligibility safeguards.

    This function performs no ranking and no selection.
    """

    if paired_rows is None:
        paired_rows = (
            build_sg_paired_development_rows()
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    expected_row_count = (
        len(SG_CANDIDATE_METHODS)
        * len(DEV_SEEDS)
    )

    if len(paired_df) != expected_row_count:
        raise RuntimeError(
            "SG paired dataframe row count mismatch: "
            f"expected={expected_row_count}, "
            f"actual={len(paired_df)}"
        )

    expected_seed_set = set(
        int(x)
        for x in DEV_SEEDS
    )

    summary_rows = []

    original_reference = None

    for method in SG_CANDIDATE_METHODS:

        sub = paired_df[
            paired_df["method"]
            == method
        ].copy()

        sub = sub.sort_values(
            "seed"
        ).reset_index(
            drop=True
        )

        actual_seed_set = set(
            int(x)
            for x
            in sub["seed"].tolist()
        )

        if actual_seed_set != expected_seed_set:
            raise RuntimeError(
                f"{method}: development seed set mismatch"
            )

        if len(sub) != len(DEV_SEEDS):
            raise RuntimeError(
                f"{method}: expected "
                f"{len(DEV_SEEDS)} paired rows, "
                f"found {len(sub)}"
            )

        this_original = {
            int(row.seed): (
                float(row.original_score),
                int(row.original_sr90),
            )
            for row
            in sub.itertuples(
                index=False
            )
        }

        if original_reference is None:
            original_reference = (
                this_original
            )

        elif (
            this_original
            != original_reference
        ):
            raise RuntimeError(
                "Original development reference differs "
                f"across SG candidates at method={method}"
            )

        window, polyorder = (
            SG_METHOD_PARAMETERS[
                method
            ]
        )

        mean_score_diff = float(
            np.mean(
                sub[
                    "score_diff_sg_minus_original"
                ].to_numpy(
                    dtype=np.float64
                )
            )
        )

        original_sr90_count = int(
            np.sum(
                sub[
                    "original_sr90"
                ].to_numpy(
                    dtype=np.int64
                )
            )
        )

        sg_sr90_count = int(
            np.sum(
                sub[
                    "sg_sr90"
                ].to_numpy(
                    dtype=np.int64
                )
            )
        )

        sr90_decrease = int(
            original_sr90_count
            - sg_sr90_count
        )

        score_safeguard_pass = bool(
            mean_score_diff
            >= SG_MEAN_SCORE_DIFF_FLOOR
        )

        sr90_safeguard_pass = bool(
            sr90_decrease
            <= SG_MAX_SR90_COUNT_DECREASE
        )

        eligible = bool(
            score_safeguard_pass
            and sr90_safeguard_pass
        )

        summary_rows.append(
            {
                "method":
                    method,

                "window":
                    int(window),

                "polyorder":
                    int(polyorder),

                "n_pairs":
                    int(len(sub)),

                "original_mean_score":
                    float(
                        np.mean(
                            sub[
                                "original_score"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "sg_mean_score":
                    float(
                        np.mean(
                            sub[
                                "sg_score"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "mean_paired_score_difference":
                    mean_score_diff,

                "score_safeguard_floor":
                    float(
                        SG_MEAN_SCORE_DIFF_FLOOR
                    ),

                "score_safeguard_pass":
                    score_safeguard_pass,

                "original_sr90_count":
                    original_sr90_count,

                "sg_sr90_count":
                    sg_sr90_count,

                "sr90_count_decrease":
                    sr90_decrease,

                "max_allowed_sr90_count_decrease":
                    int(
                        SG_MAX_SR90_COUNT_DECREASE
                    ),

                "sr90_safeguard_pass":
                    sr90_safeguard_pass,

                "eligible":
                    eligible,

                "boundary_D1_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "boundary_D1_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "D1_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "D1_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "D2_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "D2_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "D3_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "D3_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "C1_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "C1_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "C2_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "C2_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "C3_reduction_pct_mean":
                    float(
                        np.mean(
                            sub[
                                "C3_reduction_pct"
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    ),

                "common_prefix_steps_min":
                    int(
                        sub[
                            "common_prefix_steps"
                        ].min()
                    ),

                "common_prefix_steps_max":
                    int(
                        sub[
                            "common_prefix_steps"
                        ].max()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        paired_df,
        summary_df,
    )


def audit_sg_development_summary(
    paired_df=None,
    summary_df=None,
):
    """
    Mechanical audit of development summary and eligibility.

    No candidate sorting, ranking, or winner selection occurs.
    """

    if (
        paired_df is None
        or summary_df is None
    ):
        (
            paired_df,
            summary_df,
        ) = build_sg_development_summary()

    if len(summary_df) != len(
        SG_CANDIDATE_METHODS
    ):
        raise RuntimeError(
            "SG summary candidate count mismatch"
        )

    methods = (
        summary_df[
            "method"
        ].tolist()
    )

    if methods != list(
        SG_CANDIDATE_METHODS
    ):
        raise RuntimeError(
            "SG summary method order differs "
            "from frozen candidate-grid order"
        )

    if np.any(
        summary_df[
            "n_pairs"
        ].to_numpy(
            dtype=np.int64
        )
        != len(DEV_SEEDS)
    ):
        raise RuntimeError(
            "SG summary contains a candidate "
            "without exactly 20 paired seeds"
        )

    for row in summary_df.itertuples(
        index=False
    ):

        expected_score_pass = bool(
            float(
                row.mean_paired_score_difference
            )
            >= SG_MEAN_SCORE_DIFF_FLOOR
        )

        expected_sr90_pass = bool(
            int(
                row.sr90_count_decrease
            )
            <= SG_MAX_SR90_COUNT_DECREASE
        )

        expected_eligible = bool(
            expected_score_pass
            and expected_sr90_pass
        )

        if (
            bool(
                row.score_safeguard_pass
            )
            != expected_score_pass
        ):
            raise RuntimeError(
                f"{row.method}: score eligibility "
                "logic mismatch"
            )

        if (
            bool(
                row.sr90_safeguard_pass
            )
            != expected_sr90_pass
        ):
            raise RuntimeError(
                f"{row.method}: SR90 eligibility "
                "logic mismatch"
            )

        if (
            bool(
                row.eligible
            )
            != expected_eligible
        ):
            raise RuntimeError(
                f"{row.method}: combined eligibility "
                "logic mismatch"
            )

        if (
            int(
                row.original_sr90_count
            )
            < 0
            or int(
                row.original_sr90_count
            )
            > len(DEV_SEEDS)
        ):
            raise RuntimeError(
                "Original SR90 count outside "
                "valid development range"
            )

        if (
            int(
                row.sg_sr90_count
            )
            < 0
            or int(
                row.sg_sr90_count
            )
            > len(DEV_SEEDS)
        ):
            raise RuntimeError(
                "SG SR90 count outside "
                "valid development range"
            )

    time_scaling_errors = {}

    for order in (
        1,
        2,
        3,
    ):
        error = np.max(
            np.abs(
                summary_df[
                    f"D{order}_reduction_pct_mean"
                ].to_numpy(
                    dtype=np.float64
                )
                -
                summary_df[
                    f"C{order}_reduction_pct_mean"
                ].to_numpy(
                    dtype=np.float64
                )
            )
        )

        time_scaling_errors[
            order
        ] = float(
            error
        )

        if error > 1.0e-10:
            raise RuntimeError(
                "SG summary time-scaling "
                "invariance failed for "
                f"order={order}: {error}"
            )

    eligible_count = int(
        np.sum(
            summary_df[
                "eligible"
            ].to_numpy(
                dtype=bool
            )
        )
    )

    return {
        "paired_row_count":
            int(
                len(
                    paired_df
                )
            ),

        "candidate_count":
            int(
                len(
                    summary_df
                )
            ),

        "eligible_count":
            eligible_count,

        "max_C1_vs_D1_summary_error":
            time_scaling_errors[
                1
            ],

        "max_C2_vs_D2_summary_error":
            time_scaling_errors[
                2
            ],

        "max_C3_vs_D3_summary_error":
            time_scaling_errors[
                3
            ],
    }


def write_sg_development_summary_artifacts():
    """
    Atomically freeze the development descriptive summary
    and eligibility results.

    This writer deliberately does not create a selected
    candidate, rank column, winner field, or confirmatory
    launcher.
    """

    final_root = (
        SG_DEVELOPMENT_ANALYSIS_ROOT
    )

    if final_root.exists():
        raise RuntimeError(
            "SG development analysis directory "
            "already exists; refusing overwrite: "
            f"{final_root}"
        )

    paired_rows = (
        build_sg_paired_development_rows()
    )

    (
        paired_df,
        summary_df,
    ) = build_sg_development_summary(
        paired_rows
    )

    audit = (
        audit_sg_development_summary(
            paired_df,
            summary_df,
        )
    )

    parent = (
        final_root.parent
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_root = Path(
        tempfile.mkdtemp(
            prefix=".development_analysis_tmp_",
            dir=str(parent),
        )
    )

    try:

        per_seed_path = (
            tmp_root
            / "sg_development_per_seed.csv"
        )

        summary_path = (
            tmp_root
            / "sg_development_summary.csv"
        )

        eligibility_path = (
            tmp_root
            / "sg_development_eligibility.csv"
        )

        audit_path = (
            tmp_root
            / "SG_DEVELOPMENT_SUMMARY_AUDIT.txt"
        )

        manifest_path = (
            tmp_root
            / "SG_DEVELOPMENT_ANALYSIS_SHA256.txt"
        )

        method_order = {
            method: i
            for i, method
            in enumerate(
                SG_CANDIDATE_METHODS
            )
        }

        paired_out = (
            paired_df.copy()
        )

        paired_out[
            "_method_order"
        ] = paired_out[
            "method"
        ].map(
            method_order
        )

        paired_out = (
            paired_out.sort_values(
                [
                    "_method_order",
                    "seed",
                ]
            )
            .drop(
                columns=[
                    "_method_order"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        paired_out.to_csv(
            per_seed_path,
            index=False,
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        eligibility_columns = [
            "method",
            "window",
            "polyorder",
            "n_pairs",
            "mean_paired_score_difference",
            "score_safeguard_floor",
            "score_safeguard_pass",
            "original_sr90_count",
            "sg_sr90_count",
            "sr90_count_decrease",
            "max_allowed_sr90_count_decrease",
            "sr90_safeguard_pass",
            "eligible",
            "boundary_D1_reduction_pct_mean",
            "D3_reduction_pct_mean",
        ]

        summary_df[
            eligibility_columns
        ].to_csv(
            eligibility_path,
            index=False,
        )

        with audit_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "===== SG DEVELOPMENT SUMMARY AUDIT =====\n"
            )

            f.write(
                "study=Savitzky-Golay development seed42\n"
            )

            f.write(
                f"development_seed_start={min(DEV_SEEDS)}\n"
            )

            f.write(
                f"development_seed_end={max(DEV_SEEDS)}\n"
            )

            f.write(
                f"development_seed_count={len(DEV_SEEDS)}\n"
            )

            f.write(
                f"candidate_count={audit['candidate_count']}\n"
            )

            f.write(
                f"paired_row_count={audit['paired_row_count']}\n"
            )

            f.write(
                "mean_score_difference_direction="
                "SG_minus_Original\n"
            )

            f.write(
                "mean_score_difference_floor="
                f"{SG_MEAN_SCORE_DIFF_FLOOR}\n"
            )

            f.write(
                "max_sr90_count_decrease="
                f"{SG_MAX_SR90_COUNT_DECREASE}\n"
            )

            f.write(
                f"eligible_count={audit['eligible_count']}\n"
            )

            f.write(
                "selection_performed=NO\n"
            )

            f.write(
                "candidate_ranking_performed=NO\n"
            )

            f.write(
                "confirmatory_data_accessed=NO\n"
            )

            f.write(
                "post_unblind_retuning=NO\n"
            )

            f.write(
                "\n"
            )

            f.write(
                "max_C1_vs_D1_summary_error="
                f"{audit['max_C1_vs_D1_summary_error']:.17g}\n"
            )

            f.write(
                "max_C2_vs_D2_summary_error="
                f"{audit['max_C2_vs_D2_summary_error']:.17g}\n"
            )

            f.write(
                "max_C3_vs_D3_summary_error="
                f"{audit['max_C3_vs_D3_summary_error']:.17g}\n"
            )

            f.write(
                "\n===== ELIGIBILITY IN FROZEN GRID ORDER =====\n"
            )

            for row in summary_df.itertuples(
                index=False
            ):

                f.write(
                    f"{row.method}: "
                    f"window={row.window}, "
                    f"polyorder={row.polyorder}, "
                    f"mean_score_diff="
                    f"{row.mean_paired_score_difference:+.12f}, "
                    f"SR90={row.sg_sr90_count}/"
                    f"{len(DEV_SEEDS)}, "
                    f"SR90_decrease="
                    f"{row.sr90_count_decrease}, "
                    f"Boundary_D1_red="
                    f"{row.boundary_D1_reduction_pct_mean:+.12f}%, "
                    f"D3_red="
                    f"{row.D3_reduction_pct_mean:+.12f}%, "
                    f"eligible={bool(row.eligible)}\n"
                )

            f.write(
                "\nSG_DEVELOPMENT_SUMMARY_AUDIT = PASS\n"
            )

        manifest_entries = [
            (
                SG_CHECKPOINT_PATH,
                str(
                    SG_CHECKPOINT_PATH
                ),
            ),
            (
                SG_RUNNER_PATH,
                str(
                    SG_RUNNER_PATH
                ),
            ),
            (
                SG_PROTOCOL_PATH,
                str(
                    SG_PROTOCOL_PATH
                ),
            ),
            (
                SG_GRID_MANIFEST_PATH,
                str(
                    SG_GRID_MANIFEST_PATH
                ),
            ),
            (
                SG_ANALYZER_PATH,
                str(
                    SG_ANALYZER_PATH
                ),
            ),
            (
                per_seed_path,
                str(
                    final_root
                    / per_seed_path.name
                ),
            ),
            (
                summary_path,
                str(
                    final_root
                    / summary_path.name
                ),
            ),
            (
                eligibility_path,
                str(
                    final_root
                    / eligibility_path.name
                ),
            ),
            (
                audit_path,
                str(
                    final_root
                    / audit_path.name
                ),
            ),
        ]

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            for actual_path, recorded_path in (
                manifest_entries
            ):
                f.write(
                    f"{sg_sha256_of(actual_path)}  "
                    f"{recorded_path}\n"
                )

        tmp_root.rename(
            final_root
        )

    except Exception:

        if tmp_root.exists():
            import shutil

            shutil.rmtree(
                tmp_root,
                ignore_errors=True,
            )

        raise

    return {
        "output_root":
            str(
                final_root
            ),

        "paired_row_count":
            audit[
                "paired_row_count"
            ],

        "candidate_count":
            audit[
                "candidate_count"
            ],

        "eligible_count":
            audit[
                "eligible_count"
            ],
    }


# ============================================================
# Frozen SG development selection
# ============================================================

SG_SELECTION_ROOT = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_selection"
)

SG_SELECTION_SUMMARY_PATH = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_analysis/"
    "sg_development_summary.csv"
)

SG_SELECTION_ELIGIBILITY_PATH = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_analysis/"
    "sg_development_eligibility.csv"
)

SG_SELECTION_ANALYSIS_MANIFEST_PATH = Path(
    "paper_exp/savgol_baseline_seed42/"
    "development_analysis/"
    "SG_DEVELOPMENT_ANALYSIS_SHA256.txt"
)

SG_SELECTION_BOUNDARY_DECIMALS = 6


def _sg_bool_series(series):
    if series.dtype == bool:
        return series.astype(
            bool
        )

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )


def build_frozen_sg_selection():
    """
    Apply the preregistered SG selection rule exactly:

      1. Restrict to eligible candidates.
      2. Maximize mean paired own-trajectory Boundary-D1
         reduction, compared at six-decimal precision.
      3. If tied, maximize mean paired common-prefix D3
         reduction.
      4. If still tied, choose smaller window.
      5. If still tied, choose lower polyorder.

    This function accesses development artifacts only.
    """

    if not SG_SELECTION_SUMMARY_PATH.is_file():
        raise RuntimeError(
            "Development summary missing: "
            f"{SG_SELECTION_SUMMARY_PATH}"
        )

    if not SG_SELECTION_ELIGIBILITY_PATH.is_file():
        raise RuntimeError(
            "Development eligibility artifact missing: "
            f"{SG_SELECTION_ELIGIBILITY_PATH}"
        )

    summary = pd.read_csv(
        SG_SELECTION_SUMMARY_PATH
    )

    eligibility = pd.read_csv(
        SG_SELECTION_ELIGIBILITY_PATH
    )

    required_summary = {
        "method",
        "window",
        "polyorder",
        "n_pairs",
        "mean_paired_score_difference",
        "original_sr90_count",
        "sg_sr90_count",
        "sr90_count_decrease",
        "eligible",
        "boundary_D1_reduction_pct_mean",
        "D3_reduction_pct_mean",
    }

    missing = (
        required_summary
        - set(
            summary.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Development summary missing columns: "
            f"{sorted(missing)}"
        )

    if len(summary) != len(
        SG_CANDIDATE_METHODS
    ):
        raise RuntimeError(
            "Frozen SG candidate count mismatch: "
            f"expected={len(SG_CANDIDATE_METHODS)}, "
            f"actual={len(summary)}"
        )

    expected_methods = list(
        SG_CANDIDATE_METHODS
    )

    actual_methods = summary[
        "method"
    ].tolist()

    if actual_methods != expected_methods:
        raise RuntimeError(
            "Development summary candidate order "
            "differs from frozen protocol order."
        )

    if eligibility[
        "method"
    ].tolist() != expected_methods:
        raise RuntimeError(
            "Eligibility artifact candidate order "
            "differs from frozen protocol order."
        )

    summary_eligible = (
        _sg_bool_series(
            summary[
                "eligible"
            ]
        )
    )

    eligibility_eligible = (
        _sg_bool_series(
            eligibility[
                "eligible"
            ]
        )
    )

    if summary_eligible.isna().any():
        raise RuntimeError(
            "Could not parse eligibility values "
            "from development summary."
        )

    if eligibility_eligible.isna().any():
        raise RuntimeError(
            "Could not parse eligibility values "
            "from eligibility artifact."
        )

    if not np.array_equal(
        summary_eligible.to_numpy(
            dtype=bool
        ),
        eligibility_eligible.to_numpy(
            dtype=bool
        ),
    ):
        raise RuntimeError(
            "Eligibility flags disagree between "
            "frozen development artifacts."
        )

    summary = summary.copy()

    summary[
        "eligible"
    ] = summary_eligible

    summary[
        "boundary_selection_value_6dp"
    ] = [
        round(
            float(x),
            SG_SELECTION_BOUNDARY_DECIMALS,
        )
        for x in summary[
            "boundary_D1_reduction_pct_mean"
        ].tolist()
    ]

    summary[
        "boundary_best_6dp"
    ] = False

    summary[
        "d3_best_after_boundary_tie"
    ] = False

    summary[
        "selected"
    ] = False

    eligible = summary[
        summary[
            "eligible"
        ]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible SG candidate exists."
        )

    best_boundary = float(
        eligible[
            "boundary_selection_value_6dp"
        ].max()
    )

    boundary_tied = eligible[
        eligible[
            "boundary_selection_value_6dp"
        ]
        == best_boundary
    ].copy()

    summary.loc[
        summary[
            "method"
        ].isin(
            boundary_tied[
                "method"
            ]
        ),
        "boundary_best_6dp",
    ] = True

    best_d3 = float(
        boundary_tied[
            "D3_reduction_pct_mean"
        ].max()
    )

    d3_tied = boundary_tied[
        boundary_tied[
            "D3_reduction_pct_mean"
        ]
        == best_d3
    ].copy()

    summary.loc[
        summary[
            "method"
        ].isin(
            d3_tied[
                "method"
            ]
        ),
        "d3_best_after_boundary_tie",
    ] = True

    final_tied = (
        d3_tied.sort_values(
            [
                "window",
                "polyorder",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    winner = final_tied.iloc[
        0
    ]

    selected_method = str(
        winner[
            "method"
        ]
    )

    summary.loc[
        summary[
            "method"
        ]
        == selected_method,
        "selected",
    ] = True

    selected_count = int(
        summary[
            "selected"
        ].sum()
    )

    if selected_count != 1:
        raise RuntimeError(
            "Frozen SG selection did not produce "
            "exactly one selected candidate."
        )

    return (
        summary,
        {
            "selected_method":
                selected_method,

            "selected_window":
                int(
                    winner[
                        "window"
                    ]
                ),

            "selected_polyorder":
                int(
                    winner[
                        "polyorder"
                    ]
                ),

            "selected_boundary_D1_reduction_pct_mean":
                float(
                    winner[
                        "boundary_D1_reduction_pct_mean"
                    ]
                ),

            "selected_boundary_selection_value_6dp":
                float(
                    winner[
                        "boundary_selection_value_6dp"
                    ]
                ),

            "selected_D3_reduction_pct_mean":
                float(
                    winner[
                        "D3_reduction_pct_mean"
                    ]
                ),

            "selected_mean_paired_score_difference":
                float(
                    winner[
                        "mean_paired_score_difference"
                    ]
                ),

            "selected_original_sr90_count":
                int(
                    winner[
                        "original_sr90_count"
                    ]
                ),

            "selected_sg_sr90_count":
                int(
                    winner[
                        "sg_sr90_count"
                    ]
                ),

            "eligible_count":
                int(
                    eligible.shape[
                        0
                    ]
                ),

            "boundary_tie_count":
                int(
                    boundary_tied.shape[
                        0
                    ]
                ),

            "d3_tie_count":
                int(
                    d3_tied.shape[
                        0
                    ]
                ),

            "final_tie_count":
                int(
                    final_tied.shape[
                        0
                    ]
                ),
        },
    )


def audit_frozen_sg_selection(
    selection_df=None,
    selection=None,
):
    if (
        selection_df is None
        or selection is None
    ):
        (
            selection_df,
            selection,
        ) = build_frozen_sg_selection()

    selected = selection_df[
        selection_df[
            "selected"
        ]
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Selection audit expected exactly "
            "one winner."
        )

    winner = selected.iloc[
        0
    ]

    if not bool(
        winner[
            "eligible"
        ]
    ):
        raise RuntimeError(
            "Selected SG candidate is not eligible."
        )

    eligible = selection_df[
        selection_df[
            "eligible"
        ]
    ]

    best_boundary = float(
        eligible[
            "boundary_selection_value_6dp"
        ].max()
    )

    if float(
        winner[
            "boundary_selection_value_6dp"
        ]
    ) != best_boundary:
        raise RuntimeError(
            "Selected candidate does not maximize "
            "Boundary-D1 at six-decimal precision."
        )

    boundary_tied = eligible[
        eligible[
            "boundary_selection_value_6dp"
        ]
        == best_boundary
    ]

    best_d3 = float(
        boundary_tied[
            "D3_reduction_pct_mean"
        ].max()
    )

    if float(
        winner[
            "D3_reduction_pct_mean"
        ]
    ) != best_d3:
        raise RuntimeError(
            "Selected candidate fails frozen "
            "D3 tie-break."
        )

    d3_tied = boundary_tied[
        boundary_tied[
            "D3_reduction_pct_mean"
        ]
        == best_d3
    ].copy()

    expected = (
        d3_tied.sort_values(
            [
                "window",
                "polyorder",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[
            0
        ]
    )

    if str(
        winner[
            "method"
        ]
    ) != str(
        expected[
            "method"
        ]
    ):
        raise RuntimeError(
            "Selected candidate fails frozen "
            "window/polyorder tie-break."
        )

    return {
        "selected_method":
            str(
                winner[
                    "method"
                ]
            ),

        "selected_window":
            int(
                winner[
                    "window"
                ]
            ),

        "selected_polyorder":
            int(
                winner[
                    "polyorder"
                ]
            ),

        "eligible_count":
            int(
                eligible.shape[
                    0
                ]
            ),

        "boundary_tie_count":
            int(
                boundary_tied.shape[
                    0
                ]
            ),

        "d3_tie_count":
            int(
                d3_tied.shape[
                    0
                ]
            ),
    }


def write_frozen_sg_selection_artifacts():
    final_root = (
        SG_SELECTION_ROOT
    )

    if final_root.exists():
        raise RuntimeError(
            "SG selection directory already "
            "exists; refusing overwrite: "
            f"{final_root}"
        )

    (
        selection_df,
        selection,
    ) = build_frozen_sg_selection()

    audit = (
        audit_frozen_sg_selection(
            selection_df,
            selection,
        )
    )

    parent = (
        final_root.parent
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_root = Path(
        tempfile.mkdtemp(
            prefix=".development_selection_tmp_",
            dir=str(parent),
        )
    )

    try:

        candidates_path = (
            tmp_root
            / "sg_selection_candidates.csv"
        )

        selected_path = (
            tmp_root
            / "SELECTED_SG_CONFIGURATION.txt"
        )

        audit_path = (
            tmp_root
            / "SG_SELECTION_AUDIT.txt"
        )

        manifest_path = (
            tmp_root
            / "SG_SELECTION_SHA256.txt"
        )

        selection_df.to_csv(
            candidates_path,
            index=False,
        )

        with selected_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "===== FROZEN SELECTED SG CONFIGURATION =====\n"
            )

            f.write(
                f"method={selection['selected_method']}\n"
            )

            f.write(
                f"window={selection['selected_window']}\n"
            )

            f.write(
                f"polyorder={selection['selected_polyorder']}\n"
            )

            f.write(
                "savgol_mode=interp\n"
            )

            f.write(
                "action_chunk_axis=1\n"
            )

            f.write(
                "action_chunk_length=8\n"
            )

            f.write(
                "selection_boundary_precision_decimals="
                f"{SG_SELECTION_BOUNDARY_DECIMALS}\n"
            )

            f.write(
                "boundary_D1_reduction_pct_mean="
                f"{selection['selected_boundary_D1_reduction_pct_mean']:.17g}\n"
            )

            f.write(
                "boundary_selection_value_6dp="
                f"{selection['selected_boundary_selection_value_6dp']:.6f}\n"
            )

            f.write(
                "D3_reduction_pct_mean="
                f"{selection['selected_D3_reduction_pct_mean']:.17g}\n"
            )

            f.write(
                "mean_paired_score_difference="
                f"{selection['selected_mean_paired_score_difference']:.17g}\n"
            )

            f.write(
                "original_sr90_count="
                f"{selection['selected_original_sr90_count']}\n"
            )

            f.write(
                "sg_sr90_count="
                f"{selection['selected_sg_sr90_count']}\n"
            )

            f.write(
                "development_selection_frozen=YES\n"
            )

            f.write(
                "confirmatory_data_accessed=NO\n"
            )

            f.write(
                "post_unblind_retuning_permitted=NO\n"
            )

        with audit_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "===== SG FROZEN SELECTION AUDIT =====\n"
            )

            f.write(
                "selection_source="
                "frozen development analysis\n"
            )

            f.write(
                f"eligible_count={selection['eligible_count']}\n"
            )

            f.write(
                "selection_rule_1="
                "maximize eligible mean paired "
                "own-trajectory Boundary-D1 reduction\n"
            )

            f.write(
                "selection_rule_1_precision="
                "six decimal places\n"
            )

            f.write(
                "selection_rule_2="
                "maximize mean paired common-prefix "
                "D3 reduction among Boundary ties\n"
            )

            f.write(
                "selection_rule_3="
                "choose smaller window among remaining ties\n"
            )

            f.write(
                "selection_rule_4="
                "choose lower polyorder among remaining ties\n"
            )

            f.write(
                f"boundary_tie_count={selection['boundary_tie_count']}\n"
            )

            f.write(
                f"d3_tie_count={selection['d3_tie_count']}\n"
            )

            f.write(
                f"final_tie_count={selection['final_tie_count']}\n"
            )

            f.write(
                f"selected_method={selection['selected_method']}\n"
            )

            f.write(
                f"selected_window={selection['selected_window']}\n"
            )

            f.write(
                f"selected_polyorder={selection['selected_polyorder']}\n"
            )

            f.write(
                "confirmatory_data_accessed=NO\n"
            )

            f.write(
                "confirmatory_evaluation_performed=NO\n"
            )

            f.write(
                "post_unblind_retuning=NO\n"
            )

            f.write(
                "\nSG_FROZEN_SELECTION_AUDIT = PASS\n"
            )

        manifest_entries = [
            (
                SG_SELECTION_SUMMARY_PATH,
                str(
                    SG_SELECTION_SUMMARY_PATH
                ),
            ),
            (
                SG_SELECTION_ELIGIBILITY_PATH,
                str(
                    SG_SELECTION_ELIGIBILITY_PATH
                ),
            ),
            (
                SG_SELECTION_ANALYSIS_MANIFEST_PATH,
                str(
                    SG_SELECTION_ANALYSIS_MANIFEST_PATH
                ),
            ),
            (
                SG_PROTOCOL_PATH,
                str(
                    SG_PROTOCOL_PATH
                ),
            ),
            (
                SG_RUNNER_PATH,
                str(
                    SG_RUNNER_PATH
                ),
            ),
            (
                SG_ANALYZER_PATH,
                str(
                    SG_ANALYZER_PATH
                ),
            ),
            (
                candidates_path,
                str(
                    final_root
                    / candidates_path.name
                ),
            ),
            (
                selected_path,
                str(
                    final_root
                    / selected_path.name
                ),
            ),
            (
                audit_path,
                str(
                    final_root
                    / audit_path.name
                ),
            ),
        ]

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            for actual_path, recorded_path in (
                manifest_entries
            ):
                f.write(
                    f"{sg_sha256_of(actual_path)}  "
                    f"{recorded_path}\n"
                )

        tmp_root.rename(
            final_root
        )

    except Exception:

        if tmp_root.exists():
            import shutil

            shutil.rmtree(
                tmp_root,
                ignore_errors=True,
            )

        raise

    return selection
