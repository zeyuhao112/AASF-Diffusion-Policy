from pathlib import Path
import hashlib
import re
import shutil
import tempfile

import numpy as np
import pandas as pd


# ============================================================
# Frozen study definition
# ============================================================

ROOT = Path(
    "paper_exp/temporal_ensemble_fresh_seed42"
)

DEV_ROOT = ROOT / "development_grid"

TRACE_ROOT = DEV_ROOT / "traces"

SELECTION_ROOT = (
    ROOT
    / "development_selection"
)

ANALYZER = Path(
    "paper_exp/scripts/"
    "analyze_temporal_ensemble_fresh_dev_seed42_v2.py"
)

PROTOCOL = (
    ROOT
    / "TE_FRESH_PROTOCOL_FREEZE.txt"
)

# Attempt-004 is the immutable pre-selection
# 120-episode development freeze.
LOCK = (
    ROOT
    / "development_attempt_004_post_grid_accounting"
    / "FAILED_ATTEMPT_004_SHA256.txt"
)

DEV_MANIFEST = (
    DEV_ROOT
    / "TE_FRESH_DEVELOPMENT_GRID_SHA256.txt"
)

LAUNCHER = Path(
    "paper_exp/scripts/"
    "run_temporal_ensemble_fresh_dev_seed42_v4_resume.sh"
)

RUNNER = Path(
    "diffusion_policy/env_runner/"
    "pusht_keypoints_runner.py"
)

TE_CORE = Path(
    "diffusion_policy/common/"
    "temporal_ensemble.py"
)

TE_TEST = Path(
    "paper_exp/scripts/"
    "test_temporal_ensemble_alignment.py"
)

METHODS = [
    ("original", None),
    ("te_beta_0p0", 0.0),
    ("te_beta_0p5", 0.5),
    ("te_beta_1p0", 1.0),
    ("te_beta_2p0", 2.0),
    ("te_beta_4p0", 4.0),
]

TE_METHODS = METHODS[1:]

DEV_SEEDS = list(
    range(
        4800000,
        4800020,
    )
)

FORMAL_CONFIRMATORY_SEEDS = (
    4800100,
    4800149,
)

SR90_THRESHOLD = 0.90

# Preregistered task-performance safeguard.
MEAN_SCORE_DIFF_FLOOR = -0.01
MAX_SR90_COUNT_DECREASE = 1

# Frozen selection precision and tie-break.
SELECTION_BOUNDARY_DECIMALS = 6

BOOTSTRAP_SAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260816


EXPECTED_SHA256 = {
    PROTOCOL:
        "b1cedeaf355f9131b63473342ddfaf9d9209a357b2f4bb8a7b695ae1fc341564",

    LOCK:
        "ee1c31130a9347132a99624fae9e01b5c73a8eeaa2be80b00313fb4b708415ce",

    DEV_MANIFEST:
        "7c9cfae8d517bec90b3aeb4131ee384755c847e82986ac5d82f8512700bd69cc",

    LAUNCHER:
        "92f0b5712d04016d9ba78820ca568815dda2f29fbec5257bbdd2db3bc5b66ab3",

    RUNNER:
        "02e6a88caf26cf3a7509034f95a1e896a782274d37b1d9965253ac6ebf4a245d",

    TE_CORE:
        "27c3c44c265add75fb5fae5bb20c686611e765c5f6e8c376efcc27f574b4aebe",

    TE_TEST:
        "2a06570e399348609eb080778a6d4c97600236ecd56d77877d2917ff90a5e6bb",
}


# ============================================================
# Utility functions
# ============================================================

def sha256_of(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

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


def verify_development_manifest():
    if not DEV_MANIFEST.is_file():
        raise RuntimeError(
            "Development artifact manifest missing: "
            f"{DEV_MANIFEST}"
        )

    checked = 0

    with DEV_MANIFEST.open(
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
                    "Malformed development manifest "
                    f"line {line_number}: "
                    f"{line!r}"
                )

            expected = parts[0]
            path_text = parts[1].strip()

            # Support either normal sha256sum syntax or
            # binary-marker syntax.
            if path_text.startswith("*"):
                path_text = path_text[1:]

            path = Path(
                path_text
            )

            if not path.is_file():
                raise RuntimeError(
                    "Manifest-listed input missing: "
                    f"{path}"
                )

            actual = sha256_of(
                path
            )

            if actual != expected:
                raise RuntimeError(
                    "Development artifact SHA256 "
                    "verification failed:\n"
                    f"  path={path}\n"
                    f"  expected={expected}\n"
                    f"  actual={actual}"
                )

            checked += 1

    if checked == 0:
        raise RuntimeError(
            "Development artifact manifest "
            "contained no files."
        )

    return checked


def verify_confirmatory_blind():
    """
    Development selection must be completed before any
    formal confirmatory artifact exists or is inspected.
    """

    low, high = FORMAL_CONFIRMATORY_SEEDS

    forbidden_dirs = [
        ROOT / "formal_confirmatory",
        ROOT / "formal_confirmatory_analysis",
    ]

    for directory in forbidden_dirs:
        if directory.exists():
            raise RuntimeError(
                "Confirmatory-stage path exists before "
                "development selection: "
                f"{directory}"
            )

    hits = []

    for candidate in ROOT.rglob("*"):
        if not candidate.is_file():
            continue

        if candidate.suffix.lower() not in (
            ".npz",
            ".json",
        ):
            continue

        numbers = re.findall(
            r"(?<!\d)(\d{7})(?!\d)",
            candidate.name,
        )

        for number in numbers:
            seed = int(number)

            if low <= seed <= high:
                hits.append(candidate)
                break

    if hits:
        raise RuntimeError(
            "Formal confirmatory artifact detected "
            "before beta selection:\n"
            + "\n".join(
                f"  {p}"
                for p in hits[:20]
            )
        )

    return len(hits)


def finite_difference_metric(
    trajectory,
    order,
):
    """
    Canonical command-space finite-difference metric:

        mean_t || Delta^order u_t ||_2

    This development selection uses dt=1 because the
    preregistered selection criterion is the percentage
    Boundary-D1 reduction. Control-period normalization
    scales both members of a pair equally and therefore
    does not alter the percentage reduction.
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    if trajectory.ndim != 2:
        raise RuntimeError(
            "Expected executed_action shape "
            "(time, action_dim), got "
            f"{trajectory.shape}"
        )

    if trajectory.shape[0] <= order:
        raise RuntimeError(
            "Trajectory is too short for "
            f"order={order}: "
            f"length={trajectory.shape[0]}"
        )

    d = trajectory

    for _ in range(
        int(order)
    ):
        d = np.diff(
            d,
            axis=0,
        )

    return float(
        np.mean(
            np.linalg.norm(
                d,
                axis=-1,
            )
        )
    )


def boundary_d1_metric(
    executed_action,
    executed_steps_per_query,
):
    """
    Mean L2 first-difference only at REAL policy-query
    boundaries.

    If a query executes n_q controls, the next query starts
    at cumulative actual executed index sum_{i<=q} n_i.

    This intentionally uses actual execution counts rather
    than planned chunk length.
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
            "executed_steps_per_query must be 1D"
        )

    if len(counts) < 2:
        raise RuntimeError(
            "At least two executed policy queries "
            "are required for Boundary-D1."
        )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Encountered a policy query with "
            "zero actual executed controls."
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
            "actual executed action trajectory."
        )

    jumps = (
        action[starts]
        - action[starts - 1]
    )

    return float(
        np.mean(
            np.linalg.norm(
                jumps,
                axis=-1,
            )
        )
    )


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
            "Percentage reduction requires "
            "strictly positive Original metric; "
            f"got {old}"
        )

    return float(
        100.0
        * (old - new)
        / old
    )


def bootstrap_seed(
    label,
):
    digest = hashlib.sha256(
        label.encode(
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
        np.isfinite(values)
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

    remaining = (
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


def normal_trace_files(
    method,
):
    method_root = (
        TRACE_ROOT
        / method
    )

    if not method_root.is_dir():
        raise RuntimeError(
            f"Trace directory missing: "
            f"{method_root}"
        )

    files = []

    for path in sorted(
        method_root.glob(
            "test_seed_*.npz"
        )
    ):
        if path.name.endswith(
            "_overlap_diag.npz"
        ):
            continue

        files.append(
            path
        )

    return files


def seed_from_filename(
    path,
):
    match = re.fullmatch(
        r"test_seed_(\d+)\.npz",
        path.name,
    )

    if match is None:
        raise RuntimeError(
            "Unexpected normal trace filename: "
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


def load_trace_metrics(
    path,
    expected_seed,
):
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
                f"{path}: missing keys "
                f"{sorted(missing)}"
            )

        seed = int(
            np.asarray(
                z["seed"]
            ).item()
        )

        if seed != expected_seed:
            raise RuntimeError(
                f"{path}: seed mismatch: "
                f"file={expected_seed}, "
                f"payload={seed}"
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

        exact_success = int(
            score == 1.0
        )

    if not np.isfinite(
        score
    ):
        raise RuntimeError(
            f"{path}: score is not finite"
        )

    if action.ndim != 2:
        raise RuntimeError(
            f"{path}: executed_action has "
            f"shape {action.shape}"
        )

    if counts.ndim != 1:
        raise RuntimeError(
            f"{path}: executed_steps_per_query "
            "is not 1D"
        )

    if cumulative.ndim != 1:
        raise RuntimeError(
            f"{path}: cumulative_executed_steps "
            "is not 1D"
        )

    if len(counts) != len(
        cumulative
    ):
        raise RuntimeError(
            f"{path}: query-accounting lengths "
            "differ"
        )

    if len(counts) == 0:
        raise RuntimeError(
            f"{path}: no policy queries"
        )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            f"{path}: non-positive actual "
            "execution count"
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
            f"{path}: cumulative execution "
            "accounting mismatch"
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
            f"{path}: actual action length "
            "does not match per-query counts: "
            f"action={action.shape[0]}, "
            f"counts={total_from_counts}"
        )

    if actual_executed_steps != int(
        action.shape[0]
    ):
        raise RuntimeError(
            f"{path}: actual_executed_steps "
            "does not match executed_action "
            "length"
        )

    if int(
        cumulative[-1]
    ) != int(
        action.shape[0]
    ):
        raise RuntimeError(
            f"{path}: final cumulative counter "
            "does not match action length"
        )

    return {
        "seed": seed,
        "score": score,
        "sr90": int(
            score
            >= SR90_THRESHOLD
        ),
        "score_eq_1p0": (
            exact_success
        ),
        "actual_executed_steps": int(
            action.shape[0]
        ),
        "n_queries": int(
            len(counts)
        ),
        "boundary_D1": (
            boundary_d1_metric(
                action,
                counts,
            )
        ),
        "D1": (
            finite_difference_metric(
                action,
                1,
            )
        ),
        "D2": (
            finite_difference_metric(
                action,
                2,
            )
        ),
        "D3": (
            finite_difference_metric(
                action,
                3,
            )
        ),
    }


# ============================================================
# Frozen input verification
# ============================================================

if SELECTION_ROOT.exists():
    raise RuntimeError(
        "Selection directory already exists. "
        "Refusing to overwrite a potentially "
        "frozen beta selection:\n"
        f"  {SELECTION_ROOT}"
    )


verify_fixed_sha256()

confirmatory_artifact_hits = (
    verify_confirmatory_blind()
)

manifest_files_checked = (
    verify_development_manifest()
)


# ============================================================
# Exact development-seed audit
# ============================================================

expected_seed_set = set(
    DEV_SEEDS
)

method_paths = {}

for method, beta in METHODS:

    files = normal_trace_files(
        method
    )

    observed_seeds = [
        seed_from_filename(
            p
        )
        for p in files
    ]

    if len(
        observed_seeds
    ) != len(
        DEV_SEEDS
    ):
        raise RuntimeError(
            f"{method}: expected "
            f"{len(DEV_SEEDS)} normal traces, "
            f"found {len(observed_seeds)}"
        )

    if set(
        observed_seeds
    ) != expected_seed_set:
        raise RuntimeError(
            f"{method}: development seed set "
            "is not exactly "
            "4800000-4800019.\n"
            f"Observed={observed_seeds}"
        )

    if len(
        observed_seeds
    ) != len(
        set(observed_seeds)
    ):
        raise RuntimeError(
            f"{method}: duplicate seed traces"
        )

    method_paths[
        method
    ] = {
        seed_from_filename(
            p
        ): p
        for p in files
    }


# ============================================================
# Per-seed metric extraction
# ============================================================

metrics = {}

for method, beta in METHODS:

    metrics[
        method
    ] = {}

    for seed in DEV_SEEDS:

        path = method_paths[
            method
        ][
            seed
        ]

        metrics[
            method
        ][
            seed
        ] = load_trace_metrics(
            path,
            seed,
        )


# ============================================================
# Paired development table
# ============================================================

paired_rows = []

for method, beta in TE_METHODS:

    for seed in DEV_SEEDS:

        old = metrics[
            "original"
        ][
            seed
        ]

        new = metrics[
            method
        ][
            seed
        ]

        paired_rows.append(
            {
                "method": method,
                "beta": beta,
                "seed": seed,

                "original_score":
                    old["score"],
                "te_score":
                    new["score"],
                "score_diff_te_minus_original":
                    new["score"]
                    - old["score"],

                "original_sr90":
                    old["sr90"],
                "te_sr90":
                    new["sr90"],
                "sr90_diff":
                    new["sr90"]
                    - old["sr90"],

                "original_score_eq_1p0":
                    old["score_eq_1p0"],
                "te_score_eq_1p0":
                    new["score_eq_1p0"],

                "original_actual_executed_steps":
                    old[
                        "actual_executed_steps"
                    ],
                "te_actual_executed_steps":
                    new[
                        "actual_executed_steps"
                    ],

                "original_n_queries":
                    old["n_queries"],
                "te_n_queries":
                    new["n_queries"],

                "original_boundary_D1":
                    old["boundary_D1"],
                "te_boundary_D1":
                    new["boundary_D1"],

                "boundary_D1_reduction_pct":
                    percent_reduction(
                        old["boundary_D1"],
                        new["boundary_D1"],
                    ),

                "boundary_D1_absolute_reduction":
                    old["boundary_D1"]
                    - new["boundary_D1"],

                "original_D1":
                    old["D1"],
                "te_D1":
                    new["D1"],
                "D1_reduction_pct":
                    percent_reduction(
                        old["D1"],
                        new["D1"],
                    ),

                "original_D2":
                    old["D2"],
                "te_D2":
                    new["D2"],
                "D2_reduction_pct":
                    percent_reduction(
                        old["D2"],
                        new["D2"],
                    ),

                "original_D3":
                    old["D3"],
                "te_D3":
                    new["D3"],
                "D3_reduction_pct":
                    percent_reduction(
                        old["D3"],
                        new["D3"],
                    ),
            }
        )


per_seed_df = pd.DataFrame(
    paired_rows
)


# ============================================================
# Method-level descriptive summary
# ============================================================

summary_rows = []

for method, beta in METHODS:

    values = [
        metrics[
            method
        ][
            seed
        ]
        for seed in DEV_SEEDS
    ]

    scores = np.asarray(
        [
            x["score"]
            for x in values
        ],
        dtype=np.float64,
    )

    score_ci = (
        bootstrap_mean_ci(
            scores,
            f"{method}:score_mean",
        )
    )

    row = {
        "method": method,
        "beta": beta,

        "n_seeds":
            len(DEV_SEEDS),

        "score_mean":
            float(
                np.mean(
                    scores
                )
            ),

        "score_ci95_low":
            score_ci[0],

        "score_ci95_high":
            score_ci[1],

        "sr90_count":
            int(
                np.sum(
                    [
                        x["sr90"]
                        for x in values
                    ]
                )
            ),

        "score_eq_1p0_count":
            int(
                np.sum(
                    [
                        x["score_eq_1p0"]
                        for x in values
                    ]
                )
            ),

        "actual_executed_steps_mean":
            float(
                np.mean(
                    [
                        x[
                            "actual_executed_steps"
                        ]
                        for x in values
                    ]
                )
            ),

        "n_queries_mean":
            float(
                np.mean(
                    [
                        x["n_queries"]
                        for x in values
                    ]
                )
            ),

        "boundary_D1_mean":
            float(
                np.mean(
                    [
                        x["boundary_D1"]
                        for x in values
                    ]
                )
            ),

        "D1_mean":
            float(
                np.mean(
                    [
                        x["D1"]
                        for x in values
                    ]
                )
            ),

        "D2_mean":
            float(
                np.mean(
                    [
                        x["D2"]
                        for x in values
                    ]
                )
            ),

        "D3_mean":
            float(
                np.mean(
                    [
                        x["D3"]
                        for x in values
                    ]
                )
            ),
    }

    if method == "original":

        row.update(
            {
                "mean_paired_score_diff":
                    np.nan,

                "paired_score_diff_ci95_low":
                    np.nan,

                "paired_score_diff_ci95_high":
                    np.nan,

                "sr90_count_diff":
                    0,

                "boundary_D1_reduction_pct_mean":
                    np.nan,

                "boundary_D1_reduction_pct_ci95_low":
                    np.nan,

                "boundary_D1_reduction_pct_ci95_high":
                    np.nan,

                "D1_reduction_pct_mean":
                    np.nan,

                "D2_reduction_pct_mean":
                    np.nan,

                "D3_reduction_pct_mean":
                    np.nan,
            }
        )

    else:

        sub = per_seed_df[
            per_seed_df[
                "method"
            ]
            == method
        ].copy()

        score_diff = sub[
            "score_diff_te_minus_original"
        ].to_numpy(
            dtype=np.float64
        )

        boundary_red = sub[
            "boundary_D1_reduction_pct"
        ].to_numpy(
            dtype=np.float64
        )

        d1_red = sub[
            "D1_reduction_pct"
        ].to_numpy(
            dtype=np.float64
        )

        d2_red = sub[
            "D2_reduction_pct"
        ].to_numpy(
            dtype=np.float64
        )

        d3_red = sub[
            "D3_reduction_pct"
        ].to_numpy(
            dtype=np.float64
        )

        score_diff_ci = (
            bootstrap_mean_ci(
                score_diff,
                f"{method}:score_diff",
            )
        )

        boundary_ci = (
            bootstrap_mean_ci(
                boundary_red,
                f"{method}:boundary_reduction",
            )
        )

        row.update(
            {
                "mean_paired_score_diff":
                    float(
                        np.mean(
                            score_diff
                        )
                    ),

                "paired_score_diff_ci95_low":
                    score_diff_ci[0],

                "paired_score_diff_ci95_high":
                    score_diff_ci[1],

                "sr90_count_diff":
                    int(
                        np.sum(
                            sub[
                                "te_sr90"
                            ]
                        )
                        - np.sum(
                            sub[
                                "original_sr90"
                            ]
                        )
                    ),

                "boundary_D1_reduction_pct_mean":
                    float(
                        np.mean(
                            boundary_red
                        )
                    ),

                "boundary_D1_reduction_pct_ci95_low":
                    boundary_ci[0],

                "boundary_D1_reduction_pct_ci95_high":
                    boundary_ci[1],

                "D1_reduction_pct_mean":
                    float(
                        np.mean(
                            d1_red
                        )
                    ),

                "D2_reduction_pct_mean":
                    float(
                        np.mean(
                            d2_red
                        )
                    ),

                "D3_reduction_pct_mean":
                    float(
                        np.mean(
                            d3_red
                        )
                    ),
            }
        )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Mechanical preregistered beta selection
# ============================================================

original_sr90_count = int(
    summary_df.loc[
        summary_df["method"]
        == "original",
        "sr90_count",
    ].iloc[0]
)

selection_rows = []

for method, beta in TE_METHODS:

    row = summary_df.loc[
        summary_df["method"]
        == method
    ].iloc[0]

    mean_score_diff = float(
        row[
            "mean_paired_score_diff"
        ]
    )

    method_sr90_count = int(
        row[
            "sr90_count"
        ]
    )

    sr90_decrease = (
        original_sr90_count
        - method_sr90_count
    )

    score_safeguard_pass = bool(
        mean_score_diff
        >= MEAN_SCORE_DIFF_FLOOR
    )

    sr90_safeguard_pass = bool(
        sr90_decrease
        <= MAX_SR90_COUNT_DECREASE
    )

    eligible = bool(
        score_safeguard_pass
        and sr90_safeguard_pass
    )

    boundary_mean = float(
        row[
            "boundary_D1_reduction_pct_mean"
        ]
    )

    boundary_reported = round(
        boundary_mean,
        SELECTION_BOUNDARY_DECIMALS,
    )

    selection_rows.append(
        {
            "method": method,
            "beta": float(beta),

            "mean_paired_score_diff":
                mean_score_diff,

            "score_safeguard_floor":
                MEAN_SCORE_DIFF_FLOOR,

            "score_safeguard_pass":
                score_safeguard_pass,

            "original_sr90_count":
                original_sr90_count,

            "te_sr90_count":
                method_sr90_count,

            "sr90_count_decrease":
                sr90_decrease,

            "max_allowed_sr90_count_decrease":
                MAX_SR90_COUNT_DECREASE,

            "sr90_safeguard_pass":
                sr90_safeguard_pass,

            "eligible":
                eligible,

            "boundary_D1_reduction_pct_mean":
                boundary_mean,

            "boundary_selection_value_6dp":
                boundary_reported,

            "selected":
                False,
        }
    )


selection_df = pd.DataFrame(
    selection_rows
)

eligible_df = selection_df[
    selection_df[
        "eligible"
    ]
].copy()


selected_method = None
selected_beta = None

if not eligible_df.empty:

    best_reported_value = float(
        eligible_df[
            "boundary_selection_value_6dp"
        ].max()
    )

    tied = eligible_df[
        eligible_df[
            "boundary_selection_value_6dp"
        ]
        == best_reported_value
    ].copy()

    # Frozen tie-break rule: smallest beta.
    tied = tied.sort_values(
        "beta",
        ascending=True,
    )

    winner = tied.iloc[
        0
    ]

    selected_method = str(
        winner[
            "method"
        ]
    )

    selected_beta = float(
        winner[
            "beta"
        ]
    )

    selection_df.loc[
        selection_df[
            "method"
        ]
        == selected_method,
        "selected",
    ] = True


# ============================================================
# Write immutable selection artifact set
# ============================================================

tmp_dir = Path(
    tempfile.mkdtemp(
        prefix=".selection_tmp_",
        dir=str(ROOT),
    )
)

try:

    per_seed_path = (
        tmp_dir
        / "temporal_ensemble_dev_per_seed.csv"
    )

    summary_path = (
        tmp_dir
        / "temporal_ensemble_dev_summary.csv"
    )

    selection_path = (
        tmp_dir
        / "temporal_ensemble_beta_selection.csv"
    )

    frozen_beta_path = (
        tmp_dir
        / "selected_temporal_ensemble_beta.txt"
    )

    audit_path = (
        tmp_dir
        / "selection_audit.txt"
    )

    manifest_path = (
        tmp_dir
        / "selection_artifact_sha256.txt"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
        float_format="%.10g",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        float_format="%.10g",
    )

    selection_df.to_csv(
        selection_path,
        index=False,
        float_format="%.10g",
    )

    protocol_sha = sha256_of(
        PROTOCOL
    )

    lock_sha = sha256_of(
        LOCK
    )

    launcher_sha = sha256_of(
        LAUNCHER
    )

    runner_sha = sha256_of(
        RUNNER
    )

    te_core_sha = sha256_of(
        TE_CORE
    )

    te_test_sha = sha256_of(
        TE_TEST
    )

    dev_manifest_sha = sha256_of(
        DEV_MANIFEST
    )

    with frozen_beta_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TEMPORAL ENSEMBLE FROZEN "
            "DEVELOPMENT SELECTION\n"
        )

        f.write(
            "======================================\n\n"
        )

        if selected_beta is None:
            f.write(
                "selected_beta=NONE\n"
            )
            f.write(
                "selected_method=NONE\n"
            )
            f.write(
                "confirmatory_run_allowed=NO\n"
            )
            f.write(
                "development_outcome="
                "NO_BETA_PASSED_SAFEGUARD\n"
            )
        else:
            f.write(
                f"selected_beta={selected_beta:g}\n"
            )
            f.write(
                f"selected_method="
                f"{selected_method}\n"
            )
            f.write(
                "confirmatory_run_allowed=YES\n"
            )
            f.write(
                "development_outcome="
                "BETA_SELECTED\n"
            )

        f.write(
            "selection_rule=preregistered\n"
        )

        f.write(
            "selection_metric="
            "mean_paired_boundary_D1_reduction_pct\n"
        )

        f.write(
            "selection_boundary_precision_decimals="
            f"{SELECTION_BOUNDARY_DECIMALS}\n"
        )

        f.write(
            "tie_break_rule="
            "smallest_beta\n"
        )

        f.write(
            "mean_score_diff_floor="
            f"{MEAN_SCORE_DIFF_FLOOR:g}\n"
        )

        f.write(
            "max_sr90_count_decrease="
            f"{MAX_SR90_COUNT_DECREASE}\n"
        )

        f.write(
            "development_seeds="
            "4800000-4800019\n"
        )

        f.write(
            "formal_confirmatory_seeds="
            "4800100-4800149\n"
        )

        f.write(
            "confirmatory_seeds_used=NO\n"
        )

        f.write(
            f"protocol_sha256="
            f"{protocol_sha}\n"
        )

        f.write(
            f"lock_sha256="
            f"{lock_sha}\n"
        )

        f.write(
            f"development_manifest_sha256="
            f"{dev_manifest_sha}\n"
        )

        f.write(
            f"launcher_sha256="
            f"{launcher_sha}\n"
        )

        f.write(
            f"runner_sha256="
            f"{runner_sha}\n"
        )

        f.write(
            f"te_core_sha256="
            f"{te_core_sha}\n"
        )

        f.write(
            f"te_test_sha256="
            f"{te_test_sha}\n"
        )

        f.write(
            f"analysis_sha256="
            f"{sha256_of(ANALYZER)}\n"
        )

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TEMPORAL ENSEMBLE DEVELOPMENT "
            "SELECTION AUDIT\n"
        )

        f.write(
            "======================================\n\n"
        )

        f.write(
            "fixed_input_sha256=PASS\n"
        )

        f.write(
            "development_manifest_verification=PASS\n"
        )

        f.write(
            "development_manifest_files_checked="
            f"{manifest_files_checked}\n"
        )

        f.write(
            "development_seed_set_exact="
            "4800000-4800019\n"
        )

        f.write(
            "development_seed_isolation=PASS\n"
        )

        f.write(
            "confirmatory_seeds_used=NO\n"
        )

        f.write(
            "per_method_normal_trace_count=20\n"
        )

        f.write(
            "actual_execution_accounting=PASS\n"
        )

        f.write(
            "boundary_source="
            "actual_executed_query_boundaries\n"
        )

        f.write(
            "boundary_selection_metric="
            "mean_per_seed_percent_reduction\n"
        )

        f.write(
            "selection_boundary_precision_decimals="
            f"{SELECTION_BOUNDARY_DECIMALS}\n"
        )

        f.write(
            "bootstrap_samples="
            f"{BOOTSTRAP_SAMPLES}\n"
        )

        f.write(
            "bootstrap_base_seed="
            f"{BOOTSTRAP_BASE_SEED}\n"
        )

        f.write(
            "\nSafeguard rule:\n"
        )

        f.write(
            "  mean(TE - Original score) "
            ">= -0.01\n"
        )

        f.write(
            "  SR@0.90 count decrease "
            "<= 1 / 20\n"
        )

        f.write(
            "\nSelection rule:\n"
        )

        f.write(
            "  among safeguard-passing beta values,\n"
        )

        f.write(
            "  maximize mean paired "
            "Boundary-D1 reduction;\n"
        )

        f.write(
            "  ties at six decimal places "
            "select smallest beta.\n"
        )

        f.write(
            "\nEligible beta count="
            f"{len(eligible_df)}\n"
        )

        if selected_beta is None:
            f.write(
                "selected_beta=NONE\n"
            )
            f.write(
                "confirmatory_run_allowed=NO\n"
            )
        else:
            f.write(
                f"selected_beta="
                f"{selected_beta:g}\n"
            )
            f.write(
                f"selected_method="
                f"{selected_method}\n"
            )
            f.write(
                "confirmatory_run_allowed=YES\n"
            )

        f.write(
            "\nTE_DEVELOPMENT_SELECTION_AUDIT = PASS\n"
        )

    # Manifest entries use FINAL canonical paths even though
    # files are still being written in the temporary folder.
    generated_files = [
        per_seed_path,
        summary_path,
        selection_path,
        frozen_beta_path,
        audit_path,
    ]

    static_files = [
        ANALYZER,
        PROTOCOL,
        LOCK,
        DEV_MANIFEST,
        LAUNCHER,
        RUNNER,
        TE_CORE,
        TE_TEST,
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for path in static_files:
            f.write(
                f"{sha256_of(path)}  "
                f"{path}\n"
            )

        for path in generated_files:
            canonical = (
                SELECTION_ROOT
                / path.name
            )

            f.write(
                f"{sha256_of(path)}  "
                f"{canonical}\n"
            )

    # Atomic finalization. Existing selection is forbidden.
    tmp_dir.rename(
        SELECTION_ROOT
    )

except Exception:
    if tmp_dir.exists():
        shutil.rmtree(
            tmp_dir
        )
    raise


# ============================================================
# Console report
# ============================================================

print()
print(
    "===== TEMPORAL ENSEMBLE "
    "DEVELOPMENT SUMMARY ====="
)

display_cols = [
    "method",
    "beta",
    "score_mean",
    "mean_paired_score_diff",
    "sr90_count",
    "sr90_count_diff",
    "boundary_D1_reduction_pct_mean",
    "D1_reduction_pct_mean",
    "D2_reduction_pct_mean",
    "D3_reduction_pct_mean",
]

print(
    summary_df[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print(
    "===== PREREGISTERED "
    "BETA SELECTION ====="
)

selection_display_cols = [
    "method",
    "beta",
    "mean_paired_score_diff",
    "score_safeguard_pass",
    "original_sr90_count",
    "te_sr90_count",
    "sr90_count_decrease",
    "sr90_safeguard_pass",
    "eligible",
    "boundary_D1_reduction_pct_mean",
    "boundary_selection_value_6dp",
    "selected",
]

print(
    selection_df[
        selection_display_cols
    ].to_string(
        index=False
    )
)

print()

if selected_beta is None:
    print(
        "selected_beta = NONE"
    )
    print(
        "confirmatory_run_allowed = NO"
    )
else:
    print(
        f"selected_beta = "
        f"{selected_beta:g}"
    )
    print(
        f"selected_method = "
        f"{selected_method}"
    )
    print(
        "confirmatory_run_allowed = YES"
    )

print()
print(
    "TE_DEVELOPMENT_SELECTION_AUDIT = PASS"
)

if selected_beta is None:
    print(
        "TE_BETA_FROZEN = "
        "NOT_APPLICABLE"
    )
else:
    print(
        "TE_BETA_FROZEN = PASS"
    )
