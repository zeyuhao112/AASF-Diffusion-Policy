from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


# ============================================================
# Frozen formal-confirmatory study definition
# ============================================================

SG_ROOT = Path(
    "paper_exp/savgol_baseline_seed42"
)

RUN_ROOT = (
    SG_ROOT
    / "formal_confirmatory"
)

TRACE_ROOT = (
    RUN_ROOT
    / "traces"
)

ANALYSIS_ROOT = (
    SG_ROOT
    / "formal_confirmatory_analysis"
)

FORMAL_MANIFEST = (
    RUN_ROOT
    / "SG_CONFIRMATORY_ARTIFACT_SHA256.txt"
)

FORMAL_COMPLETE = (
    RUN_ROOT
    / "FORMAL_CONFIRMATORY_COMPLETE.txt"
)

PROTOCOL = (
    SG_ROOT
    / "SG_PROTOCOL_FREEZE.txt"
)

SELECTED_CONFIG = (
    SG_ROOT
    / "development_selection"
    / "SELECTED_SG_CONFIGURATION.txt"
)

SELECTION_MANIFEST = (
    SG_ROOT
    / "development_selection"
    / "SG_SELECTION_SHA256.txt"
)

FREEZE_CHAIN_MANIFEST = (
    SG_ROOT
    / "freeze_reconciliation"
    / "SG_FREEZE_CHAIN_SHA256.txt"
)

CONFIRMATORY_LAUNCHER = Path(
    "paper_exp/scripts/"
    "run_savgol_confirmatory_seed42.sh"
)

RUNNER = Path(
    "diffusion_policy/env_runner/"
    "pusht_keypoints_runner.py"
)

CHECKPOINT = Path(
    "data/checkpoints/pusht.ckpt"
)


# ============================================================
# Frozen formal arms
# ============================================================

ORIGINAL_METHOD = "original"
SG_METHOD = "w5_p3"

SELECTED_WINDOW = 5
SELECTED_POLYORDER = 3

FORMAL_SEEDS = list(
    range(
        4300100,
        4300150,
    )
)

FORMAL_SEED_SET = set(
    FORMAL_SEEDS
)

FORMAL_N = 50

SR90_THRESHOLD = 0.90

CONTROL_DT = 0.1
CONTROL_HZ = 10.0


# ============================================================
# Frozen reporting convention
#
# These are descriptive paired bootstrap intervals.
# This copies the already-existing repository convention:
# 100000 resamples and base seed 20260816.
#
# No candidate ranking, parameter selection, or retuning is
# performed by this analyzer.
# ============================================================

BOOTSTRAP_SAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260816


# ============================================================
# Frozen identities
# ============================================================

EXPECTED_SHA256 = {
    RUNNER:
        "02e6a88caf26cf3a7509034f95a1e896"
        "a782274d37b1d9965253ac6ebf4a245d",

    CHECKPOINT:
        "f804e16575e261fa0b7e981da3f67741"
        "fc8517817734320d550e43a4182bf876",

    PROTOCOL:
        "b0ba2312b017c4f62ad4216d6a8a9898"
        "9eeb1eaba8521dcd4531317aaadb7b10",

    SELECTED_CONFIG:
        "5308642320c55895246e934d6dab3b263"
        "2662004b8b0a67ac9bb1849ec70429b",

    CONFIRMATORY_LAUNCHER:
        "9d874ca22a11120d4213a4c1c49e1b41"
        "948348f2661e4ae2b51c440d5637f672",

    FORMAL_MANIFEST:
        "97e01bc2b97363e6967b630e0737e2eef"
        "790a39f09afc99c342fc04e6887cbc6",
}


# ============================================================
# SHA256 utilities
# ============================================================

def sha256_of(path):
    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

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


def verify_fixed_sha256():
    checked = 0

    for path, expected in (
        EXPECTED_SHA256.items()
    ):

        if not path.is_file():
            raise RuntimeError(
                "Frozen input missing:\n"
                f"  path={path}"
            )

        actual = sha256_of(
            path
        )

        if actual != expected:
            raise RuntimeError(
                "Frozen input SHA256 mismatch:\n"
                f"  path={path}\n"
                f"  expected={expected}\n"
                f"  actual={actual}"
            )

        checked += 1

    return checked


def verify_sha256_manifest(
    manifest,
):
    manifest = Path(
        manifest
    )

    if not manifest.is_file():
        raise RuntimeError(
            "SHA256 manifest missing:\n"
            f"  {manifest}"
        )

    checked = 0

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.rstrip(
                "\n"
            )

            if not line.strip():
                continue

            parts = line.split(
                None,
                1,
            )

            if len(parts) != 2:
                raise RuntimeError(
                    "Malformed SHA256 manifest line:\n"
                    f"  manifest={manifest}\n"
                    f"  line={line_number}\n"
                    f"  text={line!r}"
                )

            expected = parts[0]
            path_text = parts[1].strip()

            if path_text.startswith(
                "*"
            ):
                path_text = (
                    path_text[1:]
                )

            path = Path(
                path_text
            )

            if not path.is_file():
                raise RuntimeError(
                    "Manifest-listed file missing:\n"
                    f"  manifest={manifest}\n"
                    f"  path={path}"
                )

            actual = sha256_of(
                path
            )

            if actual != expected:
                raise RuntimeError(
                    "Manifest SHA256 mismatch:\n"
                    f"  manifest={manifest}\n"
                    f"  path={path}\n"
                    f"  expected={expected}\n"
                    f"  actual={actual}"
                )

            checked += 1

    if checked == 0:
        raise RuntimeError(
            "SHA256 manifest contains no entries:\n"
            f"  {manifest}"
        )

    return checked


# ============================================================
# Formal trace-path and exact-seed utilities
# ============================================================

def normal_trace_path(
    method,
    seed,
):
    return (
        TRACE_ROOT
        / str(method)
        / f"test_seed_{int(seed)}.npz"
    )


def diagnostic_trace_path(
    method,
    seed,
):
    return (
        TRACE_ROOT
        / str(method)
        / (
            f"test_seed_{int(seed)}"
            "_overlap_diag.npz"
        )
    )


def parse_normal_trace_seed(
    path,
):
    path = Path(
        path
    )

    match = re.fullmatch(
        r"test_seed_(\d+)\.npz",
        path.name,
    )

    if match is None:
        raise RuntimeError(
            "Unexpected normal trace filename:\n"
            f"  {path}"
        )

    return int(
        match.group(1)
    )


# ============================================================
# Formal method-directory audit
# ============================================================

def audit_formal_method_directories():
    if not TRACE_ROOT.is_dir():
        raise RuntimeError(
            "Formal trace root missing:\n"
            f"  {TRACE_ROOT}"
        )

    expected = {
        ORIGINAL_METHOD,
        SG_METHOD,
    }

    actual = {
        p.name
        for p in TRACE_ROOT.iterdir()
        if p.is_dir()
    }

    if actual != expected:
        raise RuntimeError(
            "Unexpected formal method directories:\n"
            f"  expected={sorted(expected)}\n"
            f"  actual={sorted(actual)}"
        )

    return sorted(
        actual
    )


# ============================================================
# Exact trace-set audit for one formal arm
# ============================================================

def audit_method_trace_set(
    method,
):
    method_dir = (
        TRACE_ROOT
        / str(method)
    )

    if not method_dir.is_dir():
        raise RuntimeError(
            "Formal method trace directory missing:\n"
            f"  {method_dir}"
        )

    normal_seeds = set()
    diagnostic_seeds = set()

    unexpected_npz = []

    for path in method_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix != ".npz":
            continue

        normal_match = re.fullmatch(
            r"test_seed_(\d+)\.npz",
            path.name,
        )

        diagnostic_match = re.fullmatch(
            r"test_seed_(\d+)"
            r"_overlap_diag\.npz",
            path.name,
        )

        if normal_match is not None:

            seed = int(
                normal_match.group(1)
            )

            normal_seeds.add(
                seed
            )

        elif diagnostic_match is not None:

            seed = int(
                diagnostic_match.group(1)
            )

            diagnostic_seeds.add(
                seed
            )

        else:
            unexpected_npz.append(
                path.name
            )

    if unexpected_npz:
        raise RuntimeError(
            "Unexpected NPZ files in formal trace "
            f"directory {method_dir}:\n"
            f"  {sorted(unexpected_npz)}"
        )

    if normal_seeds != FORMAL_SEED_SET:
        raise RuntimeError(
            "Formal normal-trace seed set mismatch:\n"
            f"  method={method}\n"
            f"  missing="
            f"{sorted(FORMAL_SEED_SET - normal_seeds)}\n"
            f"  unexpected="
            f"{sorted(normal_seeds - FORMAL_SEED_SET)}"
        )

    if diagnostic_seeds != FORMAL_SEED_SET:
        raise RuntimeError(
            "Formal diagnostic-trace seed set mismatch:\n"
            f"  method={method}\n"
            f"  missing="
            f"{sorted(FORMAL_SEED_SET - diagnostic_seeds)}\n"
            f"  unexpected="
            f"{sorted(diagnostic_seeds - FORMAL_SEED_SET)}"
        )

    if len(normal_seeds) != FORMAL_N:
        raise RuntimeError(
            "Formal normal trace count mismatch:\n"
            f"  method={method}\n"
            f"  expected={FORMAL_N}\n"
            f"  actual={len(normal_seeds)}"
        )

    if len(diagnostic_seeds) != FORMAL_N:
        raise RuntimeError(
            "Formal diagnostic trace count mismatch:\n"
            f"  method={method}\n"
            f"  expected={FORMAL_N}\n"
            f"  actual={len(diagnostic_seeds)}"
        )

    return {
        "method":
            str(method),

        "normal_count":
            len(normal_seeds),

        "diagnostic_count":
            len(diagnostic_seeds),

        "normal_seed_min":
            min(normal_seeds),

        "normal_seed_max":
            max(normal_seeds),
    }


# ============================================================
# Structural loader
#
# IMPORTANT:
# This function intentionally does NOT load or return the
# episode score.  It audits execution accounting only.
# ============================================================

def load_trace_structure(
    path,
    expected_seed,
):
    path = Path(
        path
    )

    if not path.is_file():
        raise RuntimeError(
            "Formal normal trace missing:\n"
            f"  {path}"
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
        }

        missing = (
            required
            - set(z.files)
        )

        if missing:
            raise RuntimeError(
                "Formal trace missing required keys:\n"
                f"  path={path}\n"
                f"  missing={sorted(missing)}"
            )

        seed = int(
            np.asarray(
                z["seed"]
            ).item()
        )

        if seed != int(
            expected_seed
        ):
            raise RuntimeError(
                "Formal trace seed mismatch:\n"
                f"  path={path}\n"
                f"  filename_seed={expected_seed}\n"
                f"  payload_seed={seed}"
            )

        action = np.asarray(
            z["executed_action"]
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

    if action.ndim != 2:
        raise RuntimeError(
            "executed_action must be 2D:\n"
            f"  path={path}\n"
            f"  shape={action.shape}"
        )

    if action.shape[0] <= 0:
        raise RuntimeError(
            "Empty executed action trajectory:\n"
            f"  {path}"
        )

    if action.shape[1] != 2:
        raise RuntimeError(
            "Unexpected Push-T lowdim action dimension:\n"
            f"  path={path}\n"
            f"  shape={action.shape}"
        )

    if not np.all(
        np.isfinite(
            action
        )
    ):
        raise RuntimeError(
            "Non-finite executed action detected:\n"
            f"  {path}"
        )

    if counts.ndim != 1:
        raise RuntimeError(
            "executed_steps_per_query must be 1D:\n"
            f"  {path}"
        )

    if cumulative.ndim != 1:
        raise RuntimeError(
            "cumulative_executed_steps must be 1D:\n"
            f"  {path}"
        )

    if len(counts) == 0:
        raise RuntimeError(
            "No executed policy queries:\n"
            f"  {path}"
        )

    if len(counts) != len(
        cumulative
    ):
        raise RuntimeError(
            "Execution-accounting vector lengths differ:\n"
            f"  path={path}\n"
            f"  counts={len(counts)}\n"
            f"  cumulative={len(cumulative)}"
        )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Non-positive executed controls per query:\n"
            f"  {path}"
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
            "Cumulative execution accounting mismatch:\n"
            f"  {path}"
        )

    total_from_counts = int(
        np.sum(
            counts
        )
    )

    if total_from_counts != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Executed-action length / query-count "
            "mismatch:\n"
            f"  path={path}\n"
            f"  action_steps={action.shape[0]}\n"
            f"  count_sum={total_from_counts}"
        )

    if actual_executed_steps != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "actual_executed_steps mismatch:\n"
            f"  path={path}\n"
            f"  declared={actual_executed_steps}\n"
            f"  action_steps={action.shape[0]}"
        )

    if int(
        cumulative[-1]
    ) != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Final cumulative execution count mismatch:\n"
            f"  {path}"
        )

    return {
        "seed":
            seed,

        "actual_executed_steps":
            int(action.shape[0]),

        "n_queries":
            int(len(counts)),

        "action_dim":
            int(action.shape[1]),
    }


# ============================================================
# Formal completion-marker audit
# ============================================================

def audit_formal_completion_marker():
    if not FORMAL_COMPLETE.is_file():
        raise RuntimeError(
            "Formal completion artifact missing:\n"
            f"  {FORMAL_COMPLETE}"
        )

    text = FORMAL_COMPLETE.read_text(
        encoding="utf-8"
    )

    required_lines = {
        "status=COMPLETE",
        "formal_confirmatory_unblinded=YES",
        "start_seed=4300100",
        "end_seed=4300149",
        "n_original=50",
        "n_selected_sg=50",
        "selected_method=w5_p3",
        "selected_window=5",
        "selected_polyorder=3",
        (
            "alternative_sg_configurations_"
            "evaluated=NO"
        ),
        (
            "post_unblind_retuning_"
            "permitted=NO"
        ),
    }

    present = set(
        text.splitlines()
    )

    missing = (
        required_lines
        - present
    )

    if missing:
        raise RuntimeError(
            "Formal completion artifact does not match "
            "the frozen protocol:\n"
            f"  missing={sorted(missing)}"
        )

    return True


# ============================================================
# Full formal structural / accounting audit
#
# No score is loaded and no formal effect statistic is
# calculated here.
# ============================================================

def audit_all_formal_trace_structure():
    methods = (
        audit_formal_method_directories()
    )

    method_audits = {}

    structural_records = []

    for method in methods:

        method_audits[
            method
        ] = audit_method_trace_set(
            method
        )

        for seed in FORMAL_SEEDS:

            path = normal_trace_path(
                method,
                seed,
            )

            record = load_trace_structure(
                path,
                seed,
            )

            structural_records.append(
                {
                    "method":
                        method,

                    **record,
                }
            )

    expected_total = (
        2
        * FORMAL_N
    )

    if len(
        structural_records
    ) != expected_total:
        raise RuntimeError(
            "Unexpected number of audited formal "
            "normal traces:\n"
            f"  expected={expected_total}\n"
            f"  actual={len(structural_records)}"
        )

    return {
        "methods":
            methods,

        "method_audits":
            method_audits,

        "normal_traces_audited":
            len(structural_records),
    }



# ============================================================
# Canonical smoothness metric functions
#
# These definitions are shared by the formal confirmatory
# analysis. They do not access formal traces by themselves.
# ============================================================

def finite_difference_metric(
    trajectory,
    order,
    dt=1.0,
):
    """
    Mean command-space finite-difference magnitude:

        mean_t || Delta^order u_t ||_2 / dt^order

    With dt=1 this is the raw D1/D2/D3 metric.
    With dt=CONTROL_DT this is C1/C2/C3.
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    order = int(
        order
    )

    dt = float(
        dt
    )

    if trajectory.ndim != 2:
        raise RuntimeError(
            "Expected trajectory shape "
            "(time, action_dim), got "
            f"{trajectory.shape}"
        )

    if order < 1:
        raise RuntimeError(
            "Finite-difference order must "
            "be at least 1."
        )

    if not np.isfinite(
        dt
    ):
        raise RuntimeError(
            "Finite-difference dt is not finite."
        )

    if dt <= 0.0:
        raise RuntimeError(
            "Finite-difference dt must "
            "be strictly positive."
        )

    if trajectory.shape[0] <= order:
        raise RuntimeError(
            "Trajectory is too short for "
            f"order={order}: "
            f"length={trajectory.shape[0]}"
        )

    d = trajectory

    for _ in range(
        order
    ):
        d = np.diff(
            d,
            axis=0,
        )

    d = (
        d
        / (
            dt
            ** order
        )
    )

    value = float(
        np.mean(
            np.linalg.norm(
                d,
                axis=-1,
            )
        )
    )

    if not np.isfinite(
        value
    ):
        raise RuntimeError(
            "Finite-difference metric "
            "is not finite."
        )

    return value


def boundary_d1_metric(
    executed_action,
    executed_steps_per_query,
):
    """
    Mean L2 first difference at actual policy-query
    boundaries on one trajectory.

    Boundary locations are determined from actual executed
    controls per query, not planned chunk length.
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
            "executed_action must be 2D"
        )

    if counts.ndim != 1:
        raise RuntimeError(
            "executed_steps_per_query "
            "must be 1D"
        )

    if len(
        counts
    ) < 2:
        raise RuntimeError(
            "At least two executed policy "
            "queries are required for "
            "Boundary-D1."
        )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Encountered a policy query "
            "with zero or negative actual "
            "executed controls."
        )

    if int(
        np.sum(
            counts
        )
    ) != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Boundary-D1 execution counts "
            "do not match trajectory length."
        )

    starts = np.cumsum(
        counts,
        dtype=np.int64,
    )[:-1]

    if np.any(
        starts <= 0
    ):
        raise RuntimeError(
            "Invalid query boundary index."
        )

    if np.any(
        starts >= action.shape[0]
    ):
        raise RuntimeError(
            "Query boundary lies outside "
            "actual executed trajectory."
        )

    jumps = (
        action[
            starts
        ]
        - action[
            starts - 1
        ]
    )

    value = float(
        np.mean(
            np.linalg.norm(
                jumps,
                axis=-1,
            )
        )
    )

    if not np.isfinite(
        value
    ):
        raise RuntimeError(
            "Boundary-D1 metric is "
            "not finite."
        )

    return value


def percent_reduction(
    old,
    new,
):
    old = float(
        old
    )

    new = float(
        new
    )

    if not np.isfinite(
        old
    ):
        raise RuntimeError(
            "Non-finite baseline metric."
        )

    if not np.isfinite(
        new
    ):
        raise RuntimeError(
            "Non-finite comparison metric."
        )

    if old <= 0.0:
        raise RuntimeError(
            "Percentage reduction requires "
            "strictly positive Original "
            f"metric; got {old}"
        )

    return float(
        100.0
        * (
            old
            - new
        )
        / old
    )


def paired_common_prefix_metrics(
    original_action,
    sg_action,
):
    """
    Compute paired D1/D2/D3 and C1/C2/C3 on the
    prespecified common executed prefix.

    Boundary-D1 is intentionally NOT computed here because
    it uses each arm's own actual policy-query boundaries.
    """

    original = np.asarray(
        original_action,
        dtype=np.float64,
    )

    sg = np.asarray(
        sg_action,
        dtype=np.float64,
    )

    if original.ndim != 2:
        raise RuntimeError(
            "Original executed action "
            "must be 2D."
        )

    if sg.ndim != 2:
        raise RuntimeError(
            "SG executed action "
            "must be 2D."
        )

    if (
        original.shape[1]
        != sg.shape[1]
    ):
        raise RuntimeError(
            "Original and SG action "
            "dimensions differ."
        )

    common_prefix_steps = min(
        int(
            original.shape[0]
        ),
        int(
            sg.shape[0]
        ),
    )

    if common_prefix_steps < 4:
        raise RuntimeError(
            "Common executed prefix is "
            "too short for D3: "
            f"{common_prefix_steps}"
        )

    op = original[
        :common_prefix_steps
    ]

    sp = sg[
        :common_prefix_steps
    ]

    o_d1 = finite_difference_metric(
        op,
        1,
        dt=1.0,
    )

    s_d1 = finite_difference_metric(
        sp,
        1,
        dt=1.0,
    )

    o_d2 = finite_difference_metric(
        op,
        2,
        dt=1.0,
    )

    s_d2 = finite_difference_metric(
        sp,
        2,
        dt=1.0,
    )

    o_d3 = finite_difference_metric(
        op,
        3,
        dt=1.0,
    )

    s_d3 = finite_difference_metric(
        sp,
        3,
        dt=1.0,
    )

    o_c1 = finite_difference_metric(
        op,
        1,
        dt=CONTROL_DT,
    )

    s_c1 = finite_difference_metric(
        sp,
        1,
        dt=CONTROL_DT,
    )

    o_c2 = finite_difference_metric(
        op,
        2,
        dt=CONTROL_DT,
    )

    s_c2 = finite_difference_metric(
        sp,
        2,
        dt=CONTROL_DT,
    )

    o_c3 = finite_difference_metric(
        op,
        3,
        dt=CONTROL_DT,
    )

    s_c3 = finite_difference_metric(
        sp,
        3,
        dt=CONTROL_DT,
    )

    d1_red = percent_reduction(
        o_d1,
        s_d1,
    )

    d2_red = percent_reduction(
        o_d2,
        s_d2,
    )

    d3_red = percent_reduction(
        o_d3,
        s_d3,
    )

    c1_red = percent_reduction(
        o_c1,
        s_c1,
    )

    c2_red = percent_reduction(
        o_c2,
        s_c2,
    )

    c3_red = percent_reduction(
        o_c3,
        s_c3,
    )

    if not np.isclose(
        d1_red,
        c1_red,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "C1/D1 percentage-reduction "
            "invariance failed."
        )

    if not np.isclose(
        d2_red,
        c2_red,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "C2/D2 percentage-reduction "
            "invariance failed."
        )

    if not np.isclose(
        d3_red,
        c3_red,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "C3/D3 percentage-reduction "
            "invariance failed."
        )

    return {
        "common_prefix_steps":
            common_prefix_steps,

        "original_D1":
            o_d1,
        "sg_D1":
            s_d1,
        "D1_reduction_pct":
            d1_red,

        "original_D2":
            o_d2,
        "sg_D2":
            s_d2,
        "D2_reduction_pct":
            d2_red,

        "original_D3":
            o_d3,
        "sg_D3":
            s_d3,
        "D3_reduction_pct":
            d3_red,

        "original_C1_per_s":
            o_c1,
        "sg_C1_per_s":
            s_c1,
        "C1_reduction_pct":
            c1_red,

        "original_C2_per_s2":
            o_c2,
        "sg_C2_per_s2":
            s_c2,
        "C2_reduction_pct":
            c2_red,

        "original_C3_per_s3":
            o_c3,
        "sg_C3_per_s3":
            s_c3,
        "C3_reduction_pct":
            c3_red,
    }


# ============================================================
# Formal confirmatory trace loading
#
# This block reads only the two frozen formal arms:
#   original
#   w5_p3
#
# It constructs seed-paired records but does not perform
# method-level aggregation, bootstrap inference, candidate
# selection, or retuning.
# ============================================================

FORMAL_SR90_THRESHOLD = 0.90


def load_formal_trace_record(
    method,
    seed,
):
    """
    Load one formal normal execution trace and validate
    actual-execution accounting.

    Smoothness metrics are based only on executed_action.
    """

    method = str(
        method
    )

    seed = int(
        seed
    )

    if method not in {
        "original",
        SG_METHOD,
    }:
        raise RuntimeError(
            "Unexpected formal method requested:\n"
            f"  method={method}"
        )

    if seed not in FORMAL_SEEDS:
        raise RuntimeError(
            "Requested seed is outside the frozen formal "
            "confirmatory set:\n"
            f"  seed={seed}"
        )

    path = normal_trace_path(
        method,
        seed,
    )

    if not path.is_file():
        raise RuntimeError(
            "Formal trace missing:\n"
            f"  method={method}\n"
            f"  seed={seed}\n"
            f"  path={path}"
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
        }

        missing = (
            required
            - set(z.files)
        )

        if missing:
            raise RuntimeError(
                "Formal trace is missing required keys:\n"
                f"  path={path}\n"
                f"  missing={sorted(missing)}"
            )

        payload_seed = int(
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

    if payload_seed != seed:
        raise RuntimeError(
            "Formal trace seed mismatch:\n"
            f"  path={path}\n"
            f"  expected={seed}\n"
            f"  payload={payload_seed}"
        )

    if not np.isfinite(
        score
    ):
        raise RuntimeError(
            "Formal trace score is non-finite:\n"
            f"  path={path}\n"
            f"  score={score}"
        )

    if action.ndim != 2:
        raise RuntimeError(
            "Formal executed_action must be 2D:\n"
            f"  path={path}\n"
            f"  shape={action.shape}"
        )

    if action.shape[0] < 4:
        raise RuntimeError(
            "Formal trajectory is too short for D3:\n"
            f"  path={path}\n"
            f"  steps={action.shape[0]}"
        )

    if not np.all(
        np.isfinite(
            action
        )
    ):
        raise RuntimeError(
            "Formal executed_action contains non-finite "
            "values:\n"
            f"  path={path}"
        )

    if counts.ndim != 1:
        raise RuntimeError(
            "executed_steps_per_query must be 1D:\n"
            f"  path={path}\n"
            f"  shape={counts.shape}"
        )

    if cumulative.ndim != 1:
        raise RuntimeError(
            "cumulative_executed_steps must be 1D:\n"
            f"  path={path}\n"
            f"  shape={cumulative.shape}"
        )

    if len(
        counts
    ) == 0:
        raise RuntimeError(
            "Formal trace contains no policy queries:\n"
            f"  path={path}"
        )

    if len(
        counts
    ) != len(
        cumulative
    ):
        raise RuntimeError(
            "Formal query-accounting lengths differ:\n"
            f"  path={path}\n"
            f"  counts={len(counts)}\n"
            f"  cumulative={len(cumulative)}"
        )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Formal trace contains non-positive actual "
            "execution count:\n"
            f"  path={path}"
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
            "Formal cumulative execution accounting "
            "mismatch:\n"
            f"  path={path}"
        )

    total_from_counts = int(
        np.sum(
            counts
        )
    )

    if total_from_counts != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Formal executed_action length does not match "
            "per-query execution counts:\n"
            f"  path={path}\n"
            f"  action_steps={action.shape[0]}\n"
            f"  count_sum={total_from_counts}"
        )

    if actual_executed_steps != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Formal actual_executed_steps mismatch:\n"
            f"  path={path}\n"
            f"  metadata={actual_executed_steps}\n"
            f"  action_steps={action.shape[0]}"
        )

    if int(
        cumulative[-1]
    ) != int(
        action.shape[0]
    ):
        raise RuntimeError(
            "Formal final cumulative execution counter "
            "does not equal action length:\n"
            f"  path={path}"
        )

    boundary_d1 = (
        boundary_d1_metric(
            action,
            counts,
        )
    )

    return {
        "method":
            method,

        "seed":
            seed,

        "source":
            str(path),

        "score":
            score,

        "sr90":
            int(
                score
                >= FORMAL_SR90_THRESHOLD
            ),

        "score_eq_1p0":
            int(
                score
                == 1.0
            ),

        "executed_action":
            action,

        "executed_steps_per_query":
            counts,

        "actual_executed_steps":
            int(
                action.shape[0]
            ),

        "n_queries":
            int(
                len(counts)
            ),

        "boundary_D1":
            float(
                boundary_d1
            ),
    }


def build_formal_paired_rows():
    """
    Construct exactly one frozen original-vs-selected-SG
    record for each formal confirmatory seed.

    No candidate selection is performed.
    """

    rows = []

    for seed in FORMAL_SEEDS:

        original = (
            load_formal_trace_record(
                "original",
                seed,
            )
        )

        sg = (
            load_formal_trace_record(
                SG_METHOD,
                seed,
            )
        )

        common = (
            paired_common_prefix_metrics(
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

        rows.append(
            {
                "seed":
                    int(seed),

                "original_score":
                    original[
                        "score"
                    ],

                "sg_score":
                    sg[
                        "score"
                    ],

                "score_difference":
                    (
                        sg[
                            "score"
                        ]
                        - original[
                            "score"
                        ]
                    ),

                "original_sr90":
                    original[
                        "sr90"
                    ],

                "sg_sr90":
                    sg[
                        "sr90"
                    ],

                "sr90_difference":
                    (
                        sg[
                            "sr90"
                        ]
                        - original[
                            "sr90"
                        ]
                    ),

                "original_score_eq_1p0":
                    original[
                        "score_eq_1p0"
                    ],

                "sg_score_eq_1p0":
                    sg[
                        "score_eq_1p0"
                    ],

                "original_actual_executed_steps":
                    original[
                        "actual_executed_steps"
                    ],

                "sg_actual_executed_steps":
                    sg[
                        "actual_executed_steps"
                    ],

                "original_n_queries":
                    original[
                        "n_queries"
                    ],

                "sg_n_queries":
                    sg[
                        "n_queries"
                    ],

                "common_prefix_steps":
                    common[
                        "common_prefix_steps"
                    ],

                "original_boundary_D1":
                    original[
                        "boundary_D1"
                    ],

                "sg_boundary_D1":
                    sg[
                        "boundary_D1"
                    ],

                "boundary_D1_reduction_pct":
                    boundary_reduction,

                "original_D1":
                    common[
                        "original_D1"
                    ],

                "sg_D1":
                    common[
                        "sg_D1"
                    ],

                "D1_reduction_pct":
                    common[
                        "D1_reduction_pct"
                    ],

                "original_D2":
                    common[
                        "original_D2"
                    ],

                "sg_D2":
                    common[
                        "sg_D2"
                    ],

                "D2_reduction_pct":
                    common[
                        "D2_reduction_pct"
                    ],

                "original_D3":
                    common[
                        "original_D3"
                    ],

                "sg_D3":
                    common[
                        "sg_D3"
                    ],

                "D3_reduction_pct":
                    common[
                        "D3_reduction_pct"
                    ],

                "original_C1_per_s":
                    common[
                        "original_C1_per_s"
                    ],

                "sg_C1_per_s":
                    common[
                        "sg_C1_per_s"
                    ],

                "C1_reduction_pct":
                    common[
                        "C1_reduction_pct"
                    ],

                "original_C2_per_s2":
                    common[
                        "original_C2_per_s2"
                    ],

                "sg_C2_per_s2":
                    common[
                        "sg_C2_per_s2"
                    ],

                "C2_reduction_pct":
                    common[
                        "C2_reduction_pct"
                    ],

                "original_C3_per_s3":
                    common[
                        "original_C3_per_s3"
                    ],

                "sg_C3_per_s3":
                    common[
                        "sg_C3_per_s3"
                    ],

                "C3_reduction_pct":
                    common[
                        "C3_reduction_pct"
                    ],

                "control_dt":
                    CONTROL_DT,

                "original_trace":
                    original[
                        "source"
                    ],

                "sg_trace":
                    sg[
                        "source"
                    ],
            }
        )

    return rows


def audit_formal_paired_rows(
    rows,
):
    """
    Audit exact seed pairing and metric invariants without
    performing method-level aggregation.
    """

    if len(
        rows
    ) != FORMAL_N:
        raise RuntimeError(
            "Unexpected formal paired-row count:\n"
            f"  expected={FORMAL_N}\n"
            f"  actual={len(rows)}"
        )

    seeds = [
        int(
            row[
                "seed"
            ]
        )
        for row in rows
    ]

    if len(
        set(
            seeds
        )
    ) != FORMAL_N:
        raise RuntimeError(
            "Duplicate formal paired seed detected."
        )

    if set(
        seeds
    ) != set(
        FORMAL_SEEDS
    ):
        raise RuntimeError(
            "Formal paired seed set is not exact:\n"
            f"  expected={FORMAL_SEEDS}\n"
            f"  actual={sorted(set(seeds))}"
        )

    max_c1_error = 0.0
    max_c2_error = 0.0
    max_c3_error = 0.0

    common_prefix_min = None
    common_prefix_max = None

    for row in rows:

        common_steps = int(
            row[
                "common_prefix_steps"
            ]
        )

        if common_steps < 4:
            raise RuntimeError(
                "Formal paired common prefix is too "
                "short for D3:\n"
                f"  seed={row['seed']}\n"
                f"  steps={common_steps}"
            )

        if common_steps > int(
            row[
                "original_actual_executed_steps"
            ]
        ):
            raise RuntimeError(
                "Common prefix exceeds Original "
                "trajectory:\n"
                f"  seed={row['seed']}"
            )

        if common_steps > int(
            row[
                "sg_actual_executed_steps"
            ]
        ):
            raise RuntimeError(
                "Common prefix exceeds selected-SG "
                "trajectory:\n"
                f"  seed={row['seed']}"
            )

        if common_prefix_min is None:
            common_prefix_min = (
                common_steps
            )

        if common_prefix_max is None:
            common_prefix_max = (
                common_steps
            )

        common_prefix_min = min(
            common_prefix_min,
            common_steps,
        )

        common_prefix_max = max(
            common_prefix_max,
            common_steps,
        )

        c1_error = abs(
            float(
                row[
                    "C1_reduction_pct"
                ]
            )
            - float(
                row[
                    "D1_reduction_pct"
                ]
            )
        )

        c2_error = abs(
            float(
                row[
                    "C2_reduction_pct"
                ]
            )
            - float(
                row[
                    "D2_reduction_pct"
                ]
            )
        )

        c3_error = abs(
            float(
                row[
                    "C3_reduction_pct"
                ]
            )
            - float(
                row[
                    "D3_reduction_pct"
                ]
            )
        )

        max_c1_error = max(
            max_c1_error,
            c1_error,
        )

        max_c2_error = max(
            max_c2_error,
            c2_error,
        )

        max_c3_error = max(
            max_c3_error,
            c3_error,
        )

        for name in (
            "score_difference",
            "original_boundary_D1",
            "sg_boundary_D1",
            "boundary_D1_reduction_pct",
            "D1_reduction_pct",
            "D2_reduction_pct",
            "D3_reduction_pct",
            "C1_reduction_pct",
            "C2_reduction_pct",
            "C3_reduction_pct",
        ):
            value = float(
                row[
                    name
                ]
            )

            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    "Non-finite formal paired metric:\n"
                    f"  seed={row['seed']}\n"
                    f"  metric={name}\n"
                    f"  value={value}"
                )

    tolerance = 1.0e-10

    if max_c1_error > tolerance:
        raise RuntimeError(
            "Formal C1-vs-D1 reduction invariance "
            "failed:\n"
            f"  error={max_c1_error}"
        )

    if max_c2_error > tolerance:
        raise RuntimeError(
            "Formal C2-vs-D2 reduction invariance "
            "failed:\n"
            f"  error={max_c2_error}"
        )

    if max_c3_error > tolerance:
        raise RuntimeError(
            "Formal C3-vs-D3 reduction invariance "
            "failed:\n"
            f"  error={max_c3_error}"
        )

    return {
        "paired_row_count":
            int(
                len(rows)
            ),

        "common_prefix_steps_min":
            int(
                common_prefix_min
            ),

        "common_prefix_steps_max":
            int(
                common_prefix_max
            ),

        "max_C1_vs_D1_reduction_error":
            float(
                max_c1_error
            ),

        "max_C2_vs_D2_reduction_error":
            float(
                max_c2_error
            ),

        "max_C3_vs_D3_reduction_error":
            float(
                max_c3_error
            ),
    }


# ============================================================
# Formal confirmatory descriptive aggregation
#
# The formal configuration is already frozen.  This section
# performs no configuration selection and no policy rollout.
#
# Percentage-reduction summaries are means of the 50 paired
# per-seed percentage reductions, consistent with the frozen
# development analysis.
#
# Bootstrap intervals are descriptive paired percentile
# intervals over formal seeds.
# ============================================================


def bootstrap_seed(
    label,
):
    digest = hashlib.sha256(
        str(label).encode(
            "utf-8"
        )
    ).hexdigest()

    offset = int(
        digest[:8],
        16,
    )

    return int(
        (
            BOOTSTRAP_BASE_SEED
            + offset
        )
        % (2 ** 32)
    )


def bootstrap_mean_ci(
    values,
    label,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise RuntimeError(
            "bootstrap_mean_ci expects 1D data"
        )

    if len(values) == 0:
        raise RuntimeError(
            "Cannot bootstrap empty data"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise RuntimeError(
            "Bootstrap input contains "
            "non-finite values"
        )

    rng = np.random.default_rng(
        bootstrap_seed(
            label
        )
    )

    n = len(
        values
    )

    chunk_size = 10000
    samples = []

    remaining = int(
        BOOTSTRAP_SAMPLES
    )

    while remaining > 0:

        this_n = min(
            chunk_size,
            remaining,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                this_n,
                n,
            ),
        )

        means = np.mean(
            values[
                indices
            ],
            axis=1,
        )

        samples.append(
            means
        )

        remaining -= this_n

    samples = np.concatenate(
        samples,
        axis=0,
    )

    low, high = np.quantile(
        samples,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def build_formal_confirmatory_summary(
    rows,
):
    audit_formal_paired_rows(
        rows
    )

    if len(
        rows
    ) != FORMAL_N:
        raise RuntimeError(
            "Formal summary requires exactly "
            f"{FORMAL_N} paired rows."
        )

    def arr(
        name,
    ):
        values = np.asarray(
            [
                row[name]
                for row in rows
            ],
            dtype=np.float64,
        )

        if values.shape != (
            FORMAL_N,
        ):
            raise RuntimeError(
                "Unexpected formal metric shape:\n"
                f"  metric={name}\n"
                f"  shape={values.shape}"
            )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                "Non-finite formal metric:\n"
                f"  metric={name}"
            )

        return values

    original_score = arr(
        "original_score"
    )

    sg_score = arr(
        "sg_score"
    )

    score_difference = arr(
        "score_difference"
    )

    expected_score_difference = (
        sg_score
        - original_score
    )

    if not np.allclose(
        score_difference,
        expected_score_difference,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Formal paired score-difference "
            "identity failed."
        )

    score_ci = bootstrap_mean_ci(
        score_difference,
        "sg_formal_confirmatory:"
        "score_difference",
    )

    boundary = arr(
        "boundary_D1_reduction_pct"
    )

    boundary_ci = bootstrap_mean_ci(
        boundary,
        "sg_formal_confirmatory:"
        "boundary_D1_reduction_pct",
    )

    summary = {
        "study":
            "savgol_formal_confirmatory_seed42",

        "formal_method":
            SG_METHOD,

        "selected_window":
            5,

        "selected_polyorder":
            3,

        "n_pairs":
            int(
                FORMAL_N
            ),

        "formal_seed_start":
            int(
                min(
                    FORMAL_SEEDS
                )
            ),

        "formal_seed_end":
            int(
                max(
                    FORMAL_SEEDS
                )
            ),

        "control_dt":
            float(
                CONTROL_DT
            ),

        "bootstrap_samples":
            int(
                BOOTSTRAP_SAMPLES
            ),

        "original_mean_score":
            float(
                np.mean(
                    original_score
                )
            ),

        "sg_mean_score":
            float(
                np.mean(
                    sg_score
                )
            ),

        "mean_paired_score_difference":
            float(
                np.mean(
                    score_difference
                )
            ),

        "paired_score_difference_ci95_low":
            float(
                score_ci[0]
            ),

        "paired_score_difference_ci95_high":
            float(
                score_ci[1]
            ),

        "original_sr90_count":
            int(
                np.sum(
                    arr(
                        "original_sr90"
                    )
                )
            ),

        "sg_sr90_count":
            int(
                np.sum(
                    arr(
                        "sg_sr90"
                    )
                )
            ),

        "sr90_count_difference":
            int(
                np.sum(
                    arr(
                        "sg_sr90"
                    )
                )
                - np.sum(
                    arr(
                        "original_sr90"
                    )
                )
            ),

        "original_score_eq_1p0_count":
            int(
                np.sum(
                    arr(
                        "original_score_eq_1p0"
                    )
                )
            ),

        "sg_score_eq_1p0_count":
            int(
                np.sum(
                    arr(
                        "sg_score_eq_1p0"
                    )
                )
            ),

        "common_prefix_steps_min":
            int(
                min(
                    int(
                        row[
                            "common_prefix_steps"
                        ]
                    )
                    for row in rows
                )
            ),

        "common_prefix_steps_max":
            int(
                max(
                    int(
                        row[
                            "common_prefix_steps"
                        ]
                    )
                    for row in rows
                )
            ),

        "boundary_D1_reduction_pct_mean":
            float(
                np.mean(
                    boundary
                )
            ),

        "boundary_D1_reduction_pct_ci95_low":
            float(
                boundary_ci[0]
            ),

        "boundary_D1_reduction_pct_ci95_high":
            float(
                boundary_ci[1]
            ),
    }

    metric_names = (
        "D1_reduction_pct",
        "D2_reduction_pct",
        "D3_reduction_pct",
        "C1_reduction_pct",
        "C2_reduction_pct",
        "C3_reduction_pct",
    )

    for metric in metric_names:

        values = arr(
            metric
        )

        low, high = bootstrap_mean_ci(
            values,
            (
                "sg_formal_confirmatory:"
                + metric
            ),
        )

        summary[
            metric
            + "_mean"
        ] = float(
            np.mean(
                values
            )
        )

        summary[
            metric
            + "_ci95_low"
        ] = float(
            low
        )

        summary[
            metric
            + "_ci95_high"
        ] = float(
            high
        )

    return summary


def audit_formal_confirmatory_summary(
    rows,
    summary,
):
    if int(
        summary[
            "n_pairs"
        ]
    ) != FORMAL_N:
        raise RuntimeError(
            "Formal summary paired count failed."
        )

    if str(
        summary[
            "formal_method"
        ]
    ) != SG_METHOD:
        raise RuntimeError(
            "Formal summary method identity failed."
        )

    if int(
        summary[
            "selected_window"
        ]
    ) != 5:
        raise RuntimeError(
            "Formal selected SG window changed."
        )

    if int(
        summary[
            "selected_polyorder"
        ]
    ) != 3:
        raise RuntimeError(
            "Formal selected SG polyorder changed."
        )

    seeds = {
        int(
            row[
                "seed"
            ]
        )
        for row in rows
    }

    if seeds != set(
        FORMAL_SEEDS
    ):
        raise RuntimeError(
            "Formal summary seed set is not exact."
        )

    score_difference = np.asarray(
        [
            row[
                "score_difference"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    direct_score_mean = float(
        np.mean(
            score_difference
        )
    )

    if not np.isclose(
        direct_score_mean,
        float(
            summary[
                "mean_paired_score_difference"
            ]
        ),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Formal score summary identity failed."
        )

    original_sr90 = int(
        sum(
            int(
                row[
                    "original_sr90"
                ]
            )
            for row in rows
        )
    )

    sg_sr90 = int(
        sum(
            int(
                row[
                    "sg_sr90"
                ]
            )
            for row in rows
        )
    )

    if original_sr90 != int(
        summary[
            "original_sr90_count"
        ]
    ):
        raise RuntimeError(
            "Original SR90 count audit failed."
        )

    if sg_sr90 != int(
        summary[
            "sg_sr90_count"
        ]
    ):
        raise RuntimeError(
            "Selected SG SR90 count audit failed."
        )

    if (
        sg_sr90
        - original_sr90
    ) != int(
        summary[
            "sr90_count_difference"
        ]
    ):
        raise RuntimeError(
            "SR90 count-difference audit failed."
        )

    max_c1_error = abs(
        float(
            summary[
                "C1_reduction_pct_mean"
            ]
        )
        - float(
            summary[
                "D1_reduction_pct_mean"
            ]
        )
    )

    max_c2_error = abs(
        float(
            summary[
                "C2_reduction_pct_mean"
            ]
        )
        - float(
            summary[
                "D2_reduction_pct_mean"
            ]
        )
    )

    max_c3_error = abs(
        float(
            summary[
                "C3_reduction_pct_mean"
            ]
        )
        - float(
            summary[
                "D3_reduction_pct_mean"
            ]
        )
    )

    if max(
        max_c1_error,
        max_c2_error,
        max_c3_error,
    ) > 1.0e-10:
        raise RuntimeError(
            "Formal summary time-scaling "
            "reduction invariance failed."
        )

    for key, value in summary.items():

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            if not np.isfinite(
                float(
                    value
                )
            ):
                raise RuntimeError(
                    "Non-finite formal summary value:\n"
                    f"  key={key}"
                )

    return {
        "n_pairs":
            int(
                FORMAL_N
            ),

        "max_C1_vs_D1_summary_error":
            float(
                max_c1_error
            ),

        "max_C2_vs_D2_summary_error":
            float(
                max_c2_error
            ),

        "max_C3_vs_D3_summary_error":
            float(
                max_c3_error
            ),
    }


def write_formal_confirmatory_analysis_artifacts(
    rows,
    summary,
):
    import csv
    import shutil
    import tempfile

    output_root = Path(
        "paper_exp/savgol_baseline_seed42/"
        "formal_confirmatory_analysis"
    )

    if output_root.exists():
        raise RuntimeError(
            "Formal confirmatory analysis directory "
            "already exists; refusing to overwrite:\n"
            f"  {output_root}"
        )

    parent = (
        output_root.parent
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_root = Path(
        tempfile.mkdtemp(
            prefix=(
                ".formal_confirmatory_analysis_"
            ),
            dir=str(
                parent
            ),
        )
    )

    per_seed_tmp = (
        tmp_root
        / "sg_confirmatory_per_seed.csv"
    )

    summary_tmp = (
        tmp_root
        / "sg_confirmatory_summary.csv"
    )

    audit_tmp = (
        tmp_root
        / "SG_CONFIRMATORY_ANALYSIS_AUDIT.txt"
    )

    manifest_tmp = (
        tmp_root
        / "SG_CONFIRMATORY_ANALYSIS_SHA256.txt"
    )

    try:
        if len(
            rows
        ) != FORMAL_N:
            raise RuntimeError(
                "Refusing artifact write with "
                "unexpected paired-row count."
            )

        with per_seed_tmp.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            fieldnames = list(
                rows[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    row
                )

        with summary_tmp.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            fieldnames = list(
                summary.keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerow(
                summary
            )

        analyzer_path = Path(
            __file__
        )

        protocol_path = Path(
            "paper_exp/savgol_baseline_seed42/"
            "SG_PROTOCOL_FREEZE.txt"
        )

        selected_config_path = Path(
            "paper_exp/savgol_baseline_seed42/"
            "development_selection/"
            "SELECTED_SG_CONFIGURATION.txt"
        )

        formal_manifest_path = Path(
            "paper_exp/savgol_baseline_seed42/"
            "formal_confirmatory/"
            "SG_CONFIRMATORY_ARTIFACT_SHA256.txt"
        )

        launcher_path = Path(
            "paper_exp/scripts/"
            "run_savgol_confirmatory_seed42.sh"
        )

        audit_result = (
            audit_formal_confirmatory_summary(
                rows,
                summary,
            )
        )

        audit_lines = [
            "===== SG FORMAL CONFIRMATORY ANALYSIS AUDIT =====",
            "",
            "study=savgol_formal_confirmatory_seed42",
            "formal_confirmatory_unblinded=YES",
            "formal_selected_method=w5_p3",
            "selected_window=5",
            "selected_polyorder=3",
            "alternative_sg_configurations_evaluated=NO",
            "candidate_selection_performed=NO",
            "post_unblind_retuning_performed=NO",
            "",
            f"formal_seed_start={min(FORMAL_SEEDS)}",
            f"formal_seed_end={max(FORMAL_SEEDS)}",
            f"formal_seed_count={FORMAL_N}",
            f"paired_row_count={len(rows)}",
            f"bootstrap_samples={BOOTSTRAP_SAMPLES}",
            f"control_dt={CONTROL_DT}",
            "",
            (
                "common_prefix_steps_min="
                f"{summary['common_prefix_steps_min']}"
            ),
            (
                "common_prefix_steps_max="
                f"{summary['common_prefix_steps_max']}"
            ),
            "",
            (
                "max_C1_vs_D1_summary_error="
                f"{audit_result['max_C1_vs_D1_summary_error']:.17g}"
            ),
            (
                "max_C2_vs_D2_summary_error="
                f"{audit_result['max_C2_vs_D2_summary_error']:.17g}"
            ),
            (
                "max_C3_vs_D3_summary_error="
                f"{audit_result['max_C3_vs_D3_summary_error']:.17g}"
            ),
            "",
            (
                "analyzer_sha256="
                f"{sha256_of(analyzer_path)}"
            ),
            (
                "protocol_sha256="
                f"{sha256_of(protocol_path)}"
            ),
            (
                "selected_config_sha256="
                f"{sha256_of(selected_config_path)}"
            ),
            (
                "formal_artifact_manifest_sha256="
                f"{sha256_of(formal_manifest_path)}"
            ),
            (
                "confirmatory_launcher_sha256="
                f"{sha256_of(launcher_path)}"
            ),
            "",
            "SG_FORMAL_CONFIRMATORY_ANALYSIS_AUDIT = PASS",
            "",
        ]

        audit_tmp.write_text(
            "\n".join(
                audit_lines
            ),
            encoding="utf-8",
        )

        final_per_seed = (
            output_root
            / per_seed_tmp.name
        )

        final_summary = (
            output_root
            / summary_tmp.name
        )

        final_audit = (
            output_root
            / audit_tmp.name
        )

        final_manifest = (
            output_root
            / manifest_tmp.name
        )

        manifest_entries = [
            (
                analyzer_path,
                analyzer_path,
            ),
            (
                protocol_path,
                protocol_path,
            ),
            (
                selected_config_path,
                selected_config_path,
            ),
            (
                formal_manifest_path,
                formal_manifest_path,
            ),
            (
                launcher_path,
                launcher_path,
            ),
            (
                per_seed_tmp,
                final_per_seed,
            ),
            (
                summary_tmp,
                final_summary,
            ),
            (
                audit_tmp,
                final_audit,
            ),
        ]

        manifest_lines = []

        for source, final_path in manifest_entries:

            manifest_lines.append(
                (
                    f"{sha256_of(source)}  "
                    f"{final_path}"
                )
            )

        manifest_tmp.write_text(
            "\n".join(
                manifest_lines
            )
            + "\n",
            encoding="utf-8",
        )

        shutil.move(
            str(
                tmp_root
            ),
            str(
                output_root
            ),
        )

    except Exception:
        if tmp_root.exists():
            shutil.rmtree(
                tmp_root
            )

        raise

    return {
        "output_root":
            str(
                output_root
            ),

        "per_seed":
            str(
                output_root
                / "sg_confirmatory_per_seed.csv"
            ),

        "summary":
            str(
                output_root
                / "sg_confirmatory_summary.csv"
            ),

        "audit":
            str(
                output_root
                / "SG_CONFIRMATORY_ANALYSIS_AUDIT.txt"
            ),

        "manifest":
            str(
                output_root
                / "SG_CONFIRMATORY_ANALYSIS_SHA256.txt"
            ),
    }
