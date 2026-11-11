#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(".")
TRACE_ROOT = ROOT / "paper_artifacts" / "traces"
RESULT_ROOT = ROOT / "paper_exp" / "results"
TABLE_ROOT = ROOT / "paper_exp" / "tables"

REFERENCE_RUN = "pusht_image_bsop_confirmatory_fresh50"

REFERENCE_TABLE = (
    TABLE_ROOT
    / REFERENCE_RUN
    / "pusht_image_per_seed_metrics.csv"
)

REFERENCE_CONTRAST = (
    TABLE_ROOT
    / REFERENCE_RUN
    / "pusht_image_paired_contrasts.csv"
)

REFERENCE_BINARY = (
    TABLE_ROOT
    / REFERENCE_RUN
    / "pusht_image_binary_outcomes.csv"
)

REFERENCE_MECHANISM = (
    TABLE_ROOT
    / REFERENCE_RUN
    / "pusht_image_bsop_mechanism.csv"
)


def stable_rng_seed(base_seed, label):
    digest = hashlib.sha256(
        label.encode("utf-8")
    ).digest()

    salt = int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )

    return (
        int(base_seed) + salt
    ) % (2**32)


def bootstrap_mean_ci(
    values,
    *,
    n_bootstrap,
    base_seed,
    label,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        stable_rng_seed(
            base_seed,
            label,
        )
    )

    # Chunked to keep memory modest.
    remaining = int(n_bootstrap)
    parts = []

    while remaining > 0:
        this_n = min(
            remaining,
            20000,
        )

        idx = rng.integers(
            0,
            x.size,
            size=(this_n, x.size),
        )

        parts.append(
            np.mean(
                x[idx],
                axis=1,
            )
        )

        remaining -= this_n

    boot = np.concatenate(
        parts,
        axis=0,
    )

    low, high = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return float(low), float(high)


def paired_binary_bootstrap_ci(
    baseline,
    modified,
    *,
    n_bootstrap,
    base_seed,
    label,
):
    a = np.asarray(
        baseline,
        dtype=np.float64,
    )

    b = np.asarray(
        modified,
        dtype=np.float64,
    )

    if a.shape != b.shape:
        raise RuntimeError(
            "Binary paired arrays differ in shape."
        )

    diff = b - a

    return bootstrap_mean_ci(
        diff,
        n_bootstrap=n_bootstrap,
        base_seed=base_seed,
        label=label,
    )


def finite_difference_mean(
    action,
    order,
):
    x = np.asarray(
        action,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"Action trajectory must be 2-D, got {x.shape}"
        )

    if x.shape[0] <= order:
        return np.nan

    d = np.diff(
        x,
        n=order,
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


def reduction_pct(
    baseline,
    modified,
):
    baseline = float(baseline)
    modified = float(modified)

    if not (
        np.isfinite(baseline)
        and np.isfinite(modified)
    ):
        return np.nan

    if abs(baseline) <= 1e-12:
        return np.nan

    return float(
        100.0
        * (
            baseline
            - modified
        )
        / baseline
    )


def boundary_d1_mean(
    actual_action,
    executed_steps_per_query,
):
    action = np.asarray(
        actual_action,
        dtype=np.float64,
    )

    steps = np.asarray(
        executed_steps_per_query,
        dtype=np.int64,
    )

    offset = 0
    jumps = []

    for query_index, count in enumerate(steps):
        count = int(count)

        if count < 0:
            raise RuntimeError(
                "Negative executed step count."
            )

        if (
            query_index > 0
            and count > 0
            and offset > 0
        ):
            if offset >= action.shape[0]:
                raise RuntimeError(
                    "Boundary offset outside action trajectory."
                )

            jumps.append(
                np.linalg.norm(
                    action[offset]
                    - action[offset - 1]
                )
            )

        offset += count

    if offset != action.shape[0]:
        raise RuntimeError(
            "Executed-step total does not match "
            "actual trajectory length."
        )

    if not jumps:
        return np.nan

    return float(
        np.mean(
            np.asarray(
                jumps,
                dtype=np.float64,
            )
        )
    )


def exact_success(score):
    return int(
        np.isclose(
            float(score),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    )


def sr090(score):
    return int(
        float(score) >= 0.9
    )


def load_eval_scores(run, method):
    path = (
        RESULT_ROOT
        / run
        / method
        / "eval_log.json"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing eval log: {path}"
        )

    with path.open() as f:
        data = json.load(f)

    scores = {}

    prefix = "test/sim_max_reward_"

    for key, value in data.items():
        if key.startswith(prefix):
            seed = int(
                key[len(prefix):]
            )

            scores[seed] = float(value)

    if not scores:
        raise RuntimeError(
            f"No per-seed scores in {path}"
        )

    reported_mean = float(
        data["test/mean_score"]
    )

    calculated_mean = float(
        np.mean(
            list(scores.values())
        )
    )

    if not np.isclose(
        reported_mean,
        calculated_mean,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Eval-log mean does not match "
            "per-seed score mean."
        )

    return scores


def trace_files(run, method):
    root = (
        TRACE_ROOT
        / run
        / method
    )

    files = sorted(
        root.glob("*.npz")
    )

    if not files:
        raise RuntimeError(
            f"No traces found: {root}"
        )

    return files


def load_trace_map(run, method):
    mapping = {}

    for path in trace_files(
        run,
        method,
    ):
        with np.load(
            path,
            allow_pickle=True,
        ) as d:
            seed = int(
                np.asarray(
                    d["seed"]
                ).item()
            )

            data = {
                key: np.asarray(
                    d[key]
                ).copy()
                for key in d.files
            }

        if seed in mapping:
            raise RuntimeError(
                f"Duplicate trace seed {seed}"
            )

        mapping[seed] = (
            path,
            data,
        )

    return mapping


def accounting_audit(data):
    planned = np.asarray(
        data[
            "planned_steps_per_query"
        ],
        dtype=np.int64,
    )

    executed = np.asarray(
        data[
            "executed_steps_per_query"
        ],
        dtype=np.int64,
    )

    actual = np.asarray(
        data[
            "actual_executed_action"
        ]
    )

    actual_count = int(
        np.asarray(
            data[
                "actual_executed_steps"
            ]
        ).item()
    )

    unexecuted = int(
        np.asarray(
            data[
                "unexecuted_action_steps_total"
            ]
        ).item()
    )

    executed_chunks = np.asarray(
        data["executed_action"]
    )

    if planned.shape != executed.shape:
        return False

    if np.any(executed < 0):
        return False

    if np.any(planned < executed):
        return False

    if int(np.sum(executed)) != actual_count:
        return False

    if actual.shape[0] != actual_count:
        return False

    if (
        int(np.sum(planned))
        - actual_count
        != unexecuted
    ):
        return False

    parts = []

    for i, count in enumerate(executed):
        count = int(count)

        if count > 0:
            parts.append(
                np.asarray(
                    executed_chunks[
                        i,
                        :count,
                    ]
                )
            )

    if parts:
        reconstructed = np.concatenate(
            parts,
            axis=0,
        )
    else:
        reconstructed = np.empty_like(
            actual
        )[:0]

    if not np.array_equal(
        reconstructed,
        actual,
    ):
        return False

    return True


def calculate_seed_row(
    seed,
    original_score,
    bsop_score,
    original,
    bsop,
):
    oa = np.asarray(
        original[
            "actual_executed_action"
        ]
    )

    ba = np.asarray(
        bsop[
            "actual_executed_action"
        ]
    )

    oe = np.asarray(
        original[
            "executed_steps_per_query"
        ],
        dtype=np.int64,
    )

    be = np.asarray(
        bsop[
            "executed_steps_per_query"
        ],
        dtype=np.int64,
    )

    original_steps = int(
        original[
            "actual_executed_steps"
        ]
    )

    bsop_steps = int(
        bsop[
            "actual_executed_steps"
        ]
    )

    common_steps = min(
        oa.shape[0],
        ba.shape[0],
    )

    oc = oa[:common_steps]
    bc = ba[:common_steps]

    original_D1 = finite_difference_mean(
        oa,
        1,
    )

    bsop_D1 = finite_difference_mean(
        ba,
        1,
    )

    original_D2 = finite_difference_mean(
        oa,
        2,
    )

    bsop_D2 = finite_difference_mean(
        ba,
        2,
    )

    original_D3 = finite_difference_mean(
        oa,
        3,
    )

    bsop_D3 = finite_difference_mean(
        ba,
        3,
    )

    cp_o_D1 = finite_difference_mean(
        oc,
        1,
    )

    cp_b_D1 = finite_difference_mean(
        bc,
        1,
    )

    cp_o_D2 = finite_difference_mean(
        oc,
        2,
    )

    cp_b_D2 = finite_difference_mean(
        bc,
        2,
    )

    cp_o_D3 = finite_difference_mean(
        oc,
        3,
    )

    cp_b_D3 = finite_difference_mean(
        bc,
        3,
    )

    original_boundary = (
        boundary_d1_mean(
            oa,
            oe,
        )
    )

    bsop_boundary = (
        boundary_d1_mean(
            ba,
            be,
        )
    )

    boundary_mask = np.asarray(
        bsop[
            "bsop_boundary_executed"
        ],
        dtype=bool,
    )

    raw = np.asarray(
        bsop[
            "bsop_raw_boundary_jump"
        ],
        dtype=np.float64,
    )

    projected = np.asarray(
        bsop[
            "bsop_projected_boundary_jump"
        ],
        dtype=np.float64,
    )

    correction = np.asarray(
        bsop[
            "bsop_correction"
        ],
        dtype=np.float64,
    )

    relative = np.asarray(
        bsop[
            "bsop_relative_correction"
        ],
        dtype=np.float64,
    )

    lambda_star = np.asarray(
        bsop[
            "bsop_lambda_star"
        ],
        dtype=np.float64,
    )

    lam = np.asarray(
        bsop[
            "bsop_lambda"
        ],
        dtype=np.float64,
    )

    active = np.asarray(
        bsop[
            "bsop_active"
        ],
        dtype=bool,
    )

    mask = (
        boundary_mask
        & np.isfinite(raw)
        & np.isfinite(projected)
        & np.isfinite(correction)
        & np.isfinite(relative)
        & np.isfinite(lambda_star)
        & np.isfinite(lam)
    )

    rho = float(
        np.asarray(
            bsop[
                "bsop_rho"
            ]
        ).item()
    )

    kappa = float(
        np.asarray(
            bsop[
                "bsop_kappa"
            ]
        ).item()
    )

    eta = float(
        np.asarray(
            bsop[
                "bsop_eta"
            ]
        ).item()
    )

    lambda_max = float(
        np.asarray(
            bsop[
                "bsop_lambda_max"
            ]
        ).item()
    )

    same_count = int(
        np.sum(mask)
    )

    same_active = int(
        np.sum(
            active
            & mask
        )
    )

    if same_count > 0:
        raw_mean = float(
            np.mean(raw[mask])
        )

        projected_mean = float(
            np.mean(
                projected[mask]
            )
        )

        same_reduction = reduction_pct(
            raw_mean,
            projected_mean,
        )
    else:
        raw_mean = np.nan
        projected_mean = np.nan
        same_reduction = np.nan

    boundary_violations = int(
        np.sum(
            projected[mask]
            > raw[mask] + 1e-4
        )
    )

    rho_violations = int(
        np.sum(
            correction[mask]
            > rho + 1e-4
        )
    )

    kappa_violations = int(
        np.sum(
            correction[mask]
            > (
                kappa
                * raw[mask]
                + 1e-4
            )
        )
    )

    lambda_star_violations = int(
        np.sum(
            lam[mask]
            > (
                eta
                * lambda_star[mask]
                + 1e-10
            )
        )
    )

    lambda_max_violations = int(
        np.sum(
            lam[mask]
            > lambda_max + 1e-10
        )
    )

    crn_obs = int(
        str(
            original[
                "obs_sha256"
            ][0]
        )
        ==
        str(
            bsop[
                "obs_sha256"
            ][0]
        )
    )

    crn_raw = int(
        np.array_equal(
            original[
                "raw_action"
            ][0],
            bsop[
                "raw_action"
            ][0],
        )
    )

    crn_pred = int(
        np.array_equal(
            original[
                "action_pred"
            ][0],
            bsop[
                "action_pred"
            ][0],
        )
    )

    return {
        "seed": int(seed),

        "original_score":
            float(original_score),

        "bsop_score":
            float(bsop_score),

        "score_difference":
            float(
                bsop_score
                - original_score
            ),

        "original_sr090":
            sr090(original_score),

        "bsop_sr090":
            sr090(bsop_score),

        "original_exact_success":
            exact_success(
                original_score
            ),

        "bsop_exact_success":
            exact_success(
                bsop_score
            ),

        "original_actual_executed_steps":
            original_steps,

        "bsop_actual_executed_steps":
            bsop_steps,

        "executed_step_difference":
            bsop_steps
            - original_steps,

        "original_queries":
            len(oe),

        "bsop_queries":
            len(be),

        "original_D1_mean":
            original_D1,

        "bsop_D1_mean":
            bsop_D1,

        "original_D2_mean":
            original_D2,

        "bsop_D2_mean":
            bsop_D2,

        "original_D3_mean":
            original_D3,

        "bsop_D3_mean":
            bsop_D3,

        "common_prefix_steps":
            common_steps,

        "common_prefix_original_D1":
            cp_o_D1,

        "common_prefix_bsop_D1":
            cp_b_D1,

        "common_prefix_D1_reduction_pct":
            reduction_pct(
                cp_o_D1,
                cp_b_D1,
            ),

        "common_prefix_original_D2":
            cp_o_D2,

        "common_prefix_bsop_D2":
            cp_b_D2,

        "common_prefix_D2_reduction_pct":
            reduction_pct(
                cp_o_D2,
                cp_b_D2,
            ),

        "common_prefix_original_D3":
            cp_o_D3,

        "common_prefix_bsop_D3":
            cp_b_D3,

        "common_prefix_D3_reduction_pct":
            reduction_pct(
                cp_o_D3,
                cp_b_D3,
            ),

        "original_boundary_D1_mean":
            original_boundary,

        "bsop_boundary_D1_mean":
            bsop_boundary,

        "own_trajectory_boundary_D1_reduction_pct":
            reduction_pct(
                original_boundary,
                bsop_boundary,
            ),

        "bsop_same_boundary_count":
            same_count,

        "bsop_same_boundary_active":
            same_active,

        "bsop_same_boundary_raw_mean":
            raw_mean,

        "bsop_same_boundary_projected_mean":
            projected_mean,

        "bsop_same_boundary_reduction_pct":
            same_reduction,

        "crn_initial_obs_exact":
            crn_obs,

        "crn_initial_raw_exact":
            crn_raw,

        "crn_initial_pred_exact":
            crn_pred,

        "boundary_violations":
            boundary_violations,

        "rho_violations":
            rho_violations,

        "kappa_violations":
            kappa_violations,

        "lambda_star_violations":
            lambda_star_violations,

        "lambda_max_violations":
            lambda_max_violations,
    }


PER_SEED_COLUMNS = [
    "seed",
    "original_score",
    "bsop_score",
    "score_difference",
    "original_sr090",
    "bsop_sr090",
    "original_exact_success",
    "bsop_exact_success",
    "original_actual_executed_steps",
    "bsop_actual_executed_steps",
    "executed_step_difference",
    "original_queries",
    "bsop_queries",
    "original_D1_mean",
    "bsop_D1_mean",
    "original_D2_mean",
    "bsop_D2_mean",
    "original_D3_mean",
    "bsop_D3_mean",
    "common_prefix_steps",
    "common_prefix_original_D1",
    "common_prefix_bsop_D1",
    "common_prefix_D1_reduction_pct",
    "common_prefix_original_D2",
    "common_prefix_bsop_D2",
    "common_prefix_D2_reduction_pct",
    "common_prefix_original_D3",
    "common_prefix_bsop_D3",
    "common_prefix_D3_reduction_pct",
    "original_boundary_D1_mean",
    "bsop_boundary_D1_mean",
    "own_trajectory_boundary_D1_reduction_pct",
    "bsop_same_boundary_count",
    "bsop_same_boundary_active",
    "bsop_same_boundary_raw_mean",
    "bsop_same_boundary_projected_mean",
    "bsop_same_boundary_reduction_pct",
    "crn_initial_obs_exact",
    "crn_initial_raw_exact",
    "crn_initial_pred_exact",
    "boundary_violations",
    "rho_violations",
    "kappa_violations",
    "lambda_star_violations",
    "lambda_max_violations",
]


def analyze_run_per_seed(run):
    original_scores = load_eval_scores(
        run,
        "original",
    )

    bsop_scores = load_eval_scores(
        run,
        "bsop",
    )

    original_traces = load_trace_map(
        run,
        "original",
    )

    bsop_traces = load_trace_map(
        run,
        "bsop",
    )

    seed_sets = [
        set(original_scores),
        set(bsop_scores),
        set(original_traces),
        set(bsop_traces),
    ]

    if not all(
        x == seed_sets[0]
        for x in seed_sets[1:]
    ):
        raise RuntimeError(
            "Score / trace seed sets do not match."
        )

    seeds = sorted(
        seed_sets[0]
    )

    rows = []

    accounting_ok = True
    policy_seed_ok = True

    for seed in seeds:
        _, original = (
            original_traces[seed]
        )

        _, bsop = (
            bsop_traces[seed]
        )

        if not accounting_audit(
            original
        ):
            accounting_ok = False

        if not accounting_audit(
            bsop
        ):
            accounting_ok = False

        if not bool(
            np.asarray(
                original[
                    "paired_policy_seed_enabled"
                ]
            ).item()
        ):
            policy_seed_ok = False

        if not bool(
            np.asarray(
                bsop[
                    "paired_policy_seed_enabled"
                ]
            ).item()
        ):
            policy_seed_ok = False

        if int(
            np.asarray(
                original[
                    "episode_policy_seed"
                ]
            ).item()
        ) != int(
            np.asarray(
                bsop[
                    "episode_policy_seed"
                ]
            ).item()
        ):
            policy_seed_ok = False

        rows.append(
            calculate_seed_row(
                seed,
                original_scores[seed],
                bsop_scores[seed],
                original,
                bsop,
            )
        )

    df = pd.DataFrame(
        rows,
        columns=PER_SEED_COLUMNS,
    )

    audit = {
        "trace_count_original":
            len(original_traces),

        "trace_count_bsop":
            len(bsop_traces),

        "seed_set_pass":
            True,

        "accounting_pass":
            bool(accounting_ok),

        "paired_policy_seed_pass":
            bool(policy_seed_ok),
    }

    return (
        df,
        original_traces,
        bsop_traces,
        audit,
    )


def sign_summary(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    positive = int(
        np.sum(x > 0)
    )

    negative = int(
        np.sum(x < 0)
    )

    zero = int(
        np.sum(x == 0)
    )

    discordant = (
        positive
        + negative
    )

    if discordant == 0:
        p = 1.0
    else:
        p = float(
            binomtest(
                positive,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return (
        positive,
        negative,
        zero,
        p,
    )


def build_contrasts(
    df,
    *,
    n_bootstrap,
    bootstrap_seed,
):
    metrics = [
        "score_difference",
        "executed_step_difference",
        "common_prefix_D1_reduction_pct",
        "common_prefix_D2_reduction_pct",
        "common_prefix_D3_reduction_pct",
        "own_trajectory_boundary_D1_reduction_pct",
        "bsop_same_boundary_reduction_pct",
    ]

    output_name = {
        "bsop_same_boundary_reduction_pct":
            "same_boundary_D1_reduction_pct"
    }

    rows = []

    for metric in metrics:
        values = np.asarray(
            df[metric],
            dtype=np.float64,
        )

        finite = values[
            np.isfinite(values)
        ]

        low, high = bootstrap_mean_ci(
            finite,
            n_bootstrap=n_bootstrap,
            base_seed=bootstrap_seed,
            label=(
                "continuous:"
                + metric
            ),
        )

        (
            positive,
            negative,
            zero,
            sign_p,
        ) = sign_summary(finite)

        rows.append({
            "metric":
                output_name.get(
                    metric,
                    metric,
                ),

            "n":
                len(finite),

            "mean":
                float(
                    np.mean(finite)
                ),

            "median":
                float(
                    np.median(finite)
                ),

            "ci95_low":
                low,

            "ci95_high":
                high,

            "positive":
                positive,

            "negative":
                negative,

            "zero":
                zero,

            "sign_p":
                sign_p,
        })

    return pd.DataFrame(
        rows
    )


def binary_row(
    name,
    baseline,
    modified,
    *,
    n_bootstrap,
    bootstrap_seed,
):
    a = np.asarray(
        baseline,
        dtype=np.int64,
    )

    b = np.asarray(
        modified,
        dtype=np.int64,
    )

    gain = int(
        np.sum(
            (a == 0)
            & (b == 1)
        )
    )

    loss = int(
        np.sum(
            (a == 1)
            & (b == 0)
        )
    )

    same_success = int(
        np.sum(
            (a == 1)
            & (b == 1)
        )
    )

    same_failure = int(
        np.sum(
            (a == 0)
            & (b == 0)
        )
    )

    discordant = gain + loss

    if discordant == 0:
        p = 1.0
    else:
        p = float(
            binomtest(
                gain,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    low, high = (
        paired_binary_bootstrap_ci(
            a,
            b,
            n_bootstrap=n_bootstrap,
            base_seed=bootstrap_seed,
            label=(
                "binary:"
                + name
            ),
        )
    )

    return {
        "metric":
            name,

        "baseline_successes":
            int(np.sum(a)),

        "modified_successes":
            int(np.sum(b)),

        "n":
            len(a),

        "gain":
            gain,

        "loss":
            loss,

        "same_success":
            same_success,

        "same_failure":
            same_failure,

        "rate_difference":
            float(
                np.mean(
                    b - a
                )
            ),

        "ci95_low":
            low,

        "ci95_high":
            high,

        "mcnemar_exact_p":
            p,
    }


def build_binary(
    df,
    *,
    n_bootstrap,
    bootstrap_seed,
):
    rows = [
        binary_row(
            "SR@0.90",
            df[
                "original_sr090"
            ],
            df[
                "bsop_sr090"
            ],
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        ),
        binary_row(
            "exact_success",
            df[
                "original_exact_success"
            ],
            df[
                "bsop_exact_success"
            ],
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        ),
    ]

    return pd.DataFrame(
        rows
    )


def build_mechanism(
    df,
    bsop_traces,
    *,
    n_bootstrap,
    bootstrap_seed,
):
    all_raw = []
    all_projected = []
    all_correction = []
    all_relative = []
    all_lambda = []

    active_total = 0

    for seed in sorted(
        bsop_traces
    ):
        _, d = bsop_traces[seed]

        executed = np.asarray(
            d[
                "bsop_boundary_executed"
            ],
            dtype=bool,
        )

        raw = np.asarray(
            d[
                "bsop_raw_boundary_jump"
            ],
            dtype=np.float64,
        )

        projected = np.asarray(
            d[
                "bsop_projected_boundary_jump"
            ],
            dtype=np.float64,
        )

        correction = np.asarray(
            d[
                "bsop_correction"
            ],
            dtype=np.float64,
        )

        relative = np.asarray(
            d[
                "bsop_relative_correction"
            ],
            dtype=np.float64,
        )

        lam = np.asarray(
            d[
                "bsop_lambda"
            ],
            dtype=np.float64,
        )

        active = np.asarray(
            d[
                "bsop_active"
            ],
            dtype=bool,
        )

        mask = (
            executed
            & np.isfinite(raw)
            & np.isfinite(projected)
            & np.isfinite(correction)
            & np.isfinite(relative)
            & np.isfinite(lam)
        )

        active_total += int(
            np.sum(
                active
                & mask
            )
        )

        all_raw.append(
            raw[mask]
        )

        all_projected.append(
            projected[mask]
        )

        all_correction.append(
            correction[mask]
        )

        all_relative.append(
            relative[mask]
        )

        all_lambda.append(
            lam[mask]
        )

    raw = np.concatenate(
        all_raw
    )

    projected = np.concatenate(
        all_projected
    )

    correction = np.concatenate(
        all_correction
    )

    relative = np.concatenate(
        all_relative
    )

    lam = np.concatenate(
        all_lambda
    )

    per_seed = np.asarray(
        df[
            "bsop_same_boundary_reduction_pct"
        ],
        dtype=np.float64,
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            per_seed,
            n_bootstrap=n_bootstrap,
            base_seed=bootstrap_seed,
            label=(
                "mechanism:"
                "per_seed_same_boundary"
            ),
        )
    )

    rows = [
        (
            "executed_boundaries",
            len(raw),
        ),
        (
            "active_boundaries",
            active_total,
        ),
        (
            "active_fraction",
            active_total
            / len(raw),
        ),
        (
            "raw_boundary_mean",
            np.mean(raw),
        ),
        (
            "projected_boundary_mean",
            np.mean(projected),
        ),
        (
            "pooled_same_boundary_reduction_pct",
            reduction_pct(
                np.mean(raw),
                np.mean(projected),
            ),
        ),
        (
            "per_seed_same_boundary_reduction_mean_pct",
            np.nanmean(per_seed),
        ),
        (
            "per_seed_same_boundary_reduction_median_pct",
            np.nanmedian(per_seed),
        ),
        (
            "per_seed_same_boundary_ci95_low_pct",
            ci_low,
        ),
        (
            "per_seed_same_boundary_ci95_high_pct",
            ci_high,
        ),
        (
            "per_seed_improved",
            np.sum(
                per_seed > 0
            ),
        ),
        (
            "per_seed_total",
            np.sum(
                np.isfinite(
                    per_seed
                )
            ),
        ),
        (
            "correction_mean",
            np.mean(correction),
        ),
        (
            "correction_p95",
            np.percentile(
                correction,
                95,
            ),
        ),
        (
            "correction_max",
            np.max(correction),
        ),
        (
            "relative_correction_median",
            np.median(relative),
        ),
        (
            "relative_correction_p95",
            np.percentile(
                relative,
                95,
            ),
        ),
        (
            "relative_correction_max",
            np.max(relative),
        ),
        (
            "lambda_mean",
            np.mean(lam),
        ),
        (
            "lambda_p95",
            np.percentile(
                lam,
                95,
            ),
        ),
        (
            "lambda_max_observed",
            np.max(lam),
        ),
        (
            "boundary_violations",
            np.sum(
                df[
                    "boundary_violations"
                ]
            ),
        ),
        (
            "rho_violations",
            np.sum(
                df[
                    "rho_violations"
                ]
            ),
        ),
        (
            "kappa_violations",
            np.sum(
                df[
                    "kappa_violations"
                ]
            ),
        ),
        (
            "lambda_star_violations",
            np.sum(
                df[
                    "lambda_star_violations"
                ]
            ),
        ),
        (
            "lambda_max_violations",
            np.sum(
                df[
                    "lambda_max_violations"
                ]
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "value",
        ],
    )


def regression_check_per_seed(
    calculated,
):
    if not REFERENCE_TABLE.exists():
        raise RuntimeError(
            "Seed42 reference table missing."
        )

    reference = pd.read_csv(
        REFERENCE_TABLE
    )

    a = calculated.sort_values(
        "seed"
    ).reset_index(
        drop=True
    )

    b = reference.sort_values(
        "seed"
    ).reset_index(
        drop=True
    )

    if list(a.columns) != list(b.columns):
        raise RuntimeError(
            "Seed42 per-seed column mismatch."
        )

    if a.shape != b.shape:
        raise RuntimeError(
            "Seed42 per-seed shape mismatch."
        )

    failures = []

    for col in a.columns:
        av = np.asarray(a[col])
        bv = np.asarray(b[col])

        if (
            np.issubdtype(
                av.dtype,
                np.number,
            )
            and np.issubdtype(
                bv.dtype,
                np.number,
            )
        ):
            ok = np.allclose(
                av.astype(
                    np.float64
                ),
                bv.astype(
                    np.float64
                ),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )

            if not ok:
                diff = np.nanmax(
                    np.abs(
                        av.astype(
                            np.float64
                        )
                        - bv.astype(
                            np.float64
                        )
                    )
                )

                failures.append(
                    (
                        col,
                        diff,
                    )
                )
        else:
            if not np.array_equal(
                av,
                bv,
            ):
                failures.append(
                    (
                        col,
                        "non-numeric mismatch",
                    )
                )

    if failures:
        print(
            "SEED42 PER-SEED REGRESSION FAILURES:"
        )

        for item in failures:
            print(" ", item)

        raise RuntimeError(
            "Seed42 deterministic regression failed."
        )

    print(
        "SEED42_PER_SEED_REGRESSION = PASS"
    )


def regression_check_contrast_points(
    calculated,
):
    reference = pd.read_csv(
        REFERENCE_CONTRAST
    )

    reference = reference.set_index(
        "metric"
    )

    calculated = calculated.set_index(
        "metric"
    )

    fields = [
        "n",
        "mean",
        "median",
        "positive",
        "negative",
        "zero",
        "sign_p",
    ]

    for metric in reference.index:
        if metric not in calculated.index:
            raise RuntimeError(
                f"Missing contrast metric {metric}"
            )

        for field in fields:
            a = float(
                calculated.loc[
                    metric,
                    field,
                ]
            )

            b = float(
                reference.loc[
                    metric,
                    field,
                ]
            )

            if not np.isclose(
                a,
                b,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise RuntimeError(
                    "Seed42 contrast point "
                    f"regression failed: "
                    f"{metric} {field} "
                    f"{a} != {b}"
                )

    print(
        "SEED42_CONTRAST_POINT_REGRESSION = PASS"
    )


def regression_check_binary_points(
    calculated,
):
    reference = pd.read_csv(
        REFERENCE_BINARY
    ).set_index(
        "metric"
    )

    calculated = calculated.set_index(
        "metric"
    )

    fields = [
        "baseline_successes",
        "modified_successes",
        "n",
        "gain",
        "loss",
        "same_success",
        "same_failure",
        "rate_difference",
        "mcnemar_exact_p",
    ]

    for metric in reference.index:
        for field in fields:
            a = float(
                calculated.loc[
                    metric,
                    field,
                ]
            )

            b = float(
                reference.loc[
                    metric,
                    field,
                ]
            )

            if not np.isclose(
                a,
                b,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise RuntimeError(
                    "Seed42 binary point "
                    f"regression failed: "
                    f"{metric} {field}"
                )

    print(
        "SEED42_BINARY_POINT_REGRESSION = PASS"
    )


def regression_check_mechanism_points(
    calculated,
):
    reference = pd.read_csv(
        REFERENCE_MECHANISM
    ).set_index(
        "metric"
    )

    calculated = calculated.set_index(
        "metric"
    )

    # CI values intentionally excluded because
    # the old heredoc did not preserve bootstrap
    # RNG provenance.
    excluded = {
        "per_seed_same_boundary_ci95_low_pct",
        "per_seed_same_boundary_ci95_high_pct",
    }

    for metric in reference.index:
        if metric in excluded:
            continue

        a = float(
            calculated.loc[
                metric,
                "value",
            ]
        )

        b = float(
            reference.loc[
                metric,
                "value",
            ]
        )

        if not np.isclose(
            a,
            b,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Seed42 mechanism point "
                f"regression failed: "
                f"{metric}: {a} != {b}"
            )

    print(
        "SEED42_MECHANISM_POINT_REGRESSION = PASS"
    )


def absolute_summary(df):
    return {
        "original_score":
            float(
                np.mean(
                    df[
                        "original_score"
                    ]
                )
            ),

        "bsop_score":
            float(
                np.mean(
                    df[
                        "bsop_score"
                    ]
                )
            ),

        "original_sr090":
            int(
                np.sum(
                    df[
                        "original_sr090"
                    ]
                )
            ),

        "bsop_sr090":
            int(
                np.sum(
                    df[
                        "bsop_sr090"
                    ]
                )
            ),

        "original_exact":
            int(
                np.sum(
                    df[
                        "original_exact_success"
                    ]
                )
            ),

        "bsop_exact":
            int(
                np.sum(
                    df[
                        "bsop_exact_success"
                    ]
                )
            ),

        "original_steps":
            float(
                np.mean(
                    df[
                        "original_actual_executed_steps"
                    ]
                )
            ),

        "bsop_steps":
            float(
                np.mean(
                    df[
                        "bsop_actual_executed_steps"
                    ]
                )
            ),

        "original_D1":
            float(
                np.mean(
                    df[
                        "original_D1_mean"
                    ]
                )
            ),

        "bsop_D1":
            float(
                np.mean(
                    df[
                        "bsop_D1_mean"
                    ]
                )
            ),

        "original_D2":
            float(
                np.mean(
                    df[
                        "original_D2_mean"
                    ]
                )
            ),

        "bsop_D2":
            float(
                np.mean(
                    df[
                        "bsop_D2_mean"
                    ]
                )
            ),

        "original_D3":
            float(
                np.mean(
                    df[
                        "original_D3_mean"
                    ]
                )
            ),

        "bsop_D3":
            float(
                np.mean(
                    df[
                        "bsop_D3_mean"
                    ]
                )
            ),

        "original_boundary":
            float(
                np.mean(
                    df[
                        "original_boundary_D1_mean"
                    ]
                )
            ),

        "bsop_boundary":
            float(
                np.mean(
                    df[
                        "bsop_boundary_D1_mean"
                    ]
                )
            ),
    }


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def write_summary(
    path,
    run,
    df,
    contrasts,
    binary,
    mechanism,
    audit,
    *,
    n_bootstrap,
    bootstrap_seed,
):
    a = absolute_summary(df)

    contrast_map = {
        row["metric"]: row
        for row in contrasts.to_dict(
            orient="records"
        )
    }

    mechanism_map = {
        row["metric"]: row["value"]
        for row in mechanism.to_dict(
            orient="records"
        )
    }

    lines = []

    lines.append(
        "===== PUSHT IMAGE BSOP ANALYSIS ====="
    )
    lines.append(
        f"run = {run}"
    )
    lines.append(
        f"bootstrap_resamples = {n_bootstrap}"
    )
    lines.append(
        f"bootstrap_seed = {bootstrap_seed}"
    )
    lines.append("")

    lines.append(
        "===== ABSOLUTE SUMMARY ====="
    )

    lines.append(
        "method             score     "
        "SR@0.90     exact       steps"
        "            D1            D2"
        "            D3      boundaryD1"
    )

    lines.append(
        f"original        "
        f"{a['original_score']:.6f}    "
        f"{a['original_sr090']:2d}/50    "
        f"{a['original_exact']:2d}/50    "
        f"{a['original_steps']:8.2f}    "
        f"{a['original_D1']:10.6f}    "
        f"{a['original_D2']:10.6f}    "
        f"{a['original_D3']:10.6f}    "
        f"{a['original_boundary']:12.6f}"
    )

    lines.append(
        f"bsop            "
        f"{a['bsop_score']:.6f}    "
        f"{a['bsop_sr090']:2d}/50    "
        f"{a['bsop_exact']:2d}/50    "
        f"{a['bsop_steps']:8.2f}    "
        f"{a['bsop_D1']:10.6f}    "
        f"{a['bsop_D2']:10.6f}    "
        f"{a['bsop_D3']:10.6f}    "
        f"{a['bsop_boundary']:12.6f}"
    )

    lines.append("")
    lines.append(
        "===== PAIRED CONTINUOUS CONTRASTS ====="
    )

    for row in contrasts.to_dict(
        orient="records"
    ):
        lines.append(
            f"{row['metric']}: "
            f"mean={row['mean']:+.6f}, "
            f"95% CI=[{row['ci95_low']:+.6f}, "
            f"{row['ci95_high']:+.6f}], "
            f"median={row['median']:+.6f}, "
            f"positive/negative/zero="
            f"{int(row['positive'])}/"
            f"{int(row['negative'])}/"
            f"{int(row['zero'])}, "
            f"sign_p={row['sign_p']:.9g}"
        )

    lines.append("")
    lines.append(
        "===== PAIRED BINARY OUTCOMES ====="
    )

    for row in binary.to_dict(
        orient="records"
    ):
        lines.append(
            f"{row['metric']}: "
            f"{int(row['baseline_successes'])}/50 -> "
            f"{int(row['modified_successes'])}/50"
        )

        lines.append(
            "  gain/loss/same-success/same-failure = "
            f"{int(row['gain'])}/"
            f"{int(row['loss'])}/"
            f"{int(row['same_success'])}/"
            f"{int(row['same_failure'])}"
        )

        lines.append(
            "  paired rate difference = "
            f"{100.0 * row['rate_difference']:+.2f} pp, "
            "95% bootstrap CI = "
            f"[{100.0 * row['ci95_low']:+.2f}, "
            f"{100.0 * row['ci95_high']:+.2f}] pp"
        )

        lines.append(
            "  exact McNemar/binomial p = "
            f"{row['mcnemar_exact_p']:.9g}"
        )

    lines.append("")
    lines.append(
        "===== BSOP SAME-BOUNDARY MECHANISM ====="
    )

    lines.append(
        "Executed boundaries: "
        f"{int(mechanism_map['executed_boundaries'])}"
    )

    lines.append(
        "Active transitions: "
        f"{100.0 * mechanism_map['active_fraction']:.2f}%"
    )

    lines.append(
        "Raw/projected boundary mean: "
        f"{mechanism_map['raw_boundary_mean']:.9f} / "
        f"{mechanism_map['projected_boundary_mean']:.9f}"
    )

    lines.append(
        "Pooled same-boundary reduction: "
        f"{mechanism_map['pooled_same_boundary_reduction_pct']:.4f}%"
    )

    lines.append(
        "Per-seed same-boundary reduction: "
        f"mean="
        f"{mechanism_map['per_seed_same_boundary_reduction_mean_pct']:.4f}%, "
        "95% CI=["
        f"{mechanism_map['per_seed_same_boundary_ci95_low_pct']:.4f}, "
        f"{mechanism_map['per_seed_same_boundary_ci95_high_pct']:.4f}]%, "
        "improved="
        f"{int(mechanism_map['per_seed_improved'])}/"
        f"{int(mechanism_map['per_seed_total'])}"
    )

    lines.append(
        "Violations "
        "(boundary/rho/kappa/lambda-star/lambda-max): "
        f"{int(mechanism_map['boundary_violations'])}/"
        f"{int(mechanism_map['rho_violations'])}/"
        f"{int(mechanism_map['kappa_violations'])}/"
        f"{int(mechanism_map['lambda_star_violations'])}/"
        f"{int(mechanism_map['lambda_max_violations'])}"
    )

    lines.append("")
    lines.append(
        "===== AUDIT ====="
    )

    crn_pass = bool(
        np.all(
            df[
                [
                    "crn_initial_obs_exact",
                    "crn_initial_raw_exact",
                    "crn_initial_pred_exact",
                ]
            ].to_numpy()
            == 1
        )
    )

    constraint_pass = bool(
        np.sum(
            df[
                [
                    "boundary_violations",
                    "rho_violations",
                    "kappa_violations",
                    "lambda_star_violations",
                    "lambda_max_violations",
                ]
            ].to_numpy()
        )
        == 0
    )

    lines.append(
        "FORMAL_TRACE_AUDIT = "
        + (
            "PASS"
            if (
                audit[
                    "trace_count_original"
                ] == 50
                and audit[
                    "trace_count_bsop"
                ] == 50
                and audit[
                    "seed_set_pass"
                ]
            )
            else "FAIL"
        )
    )

    lines.append(
        "CRN_INITIAL_QUERY_AUDIT = "
        + (
            "PASS"
            if crn_pass
            else "FAIL"
        )
    )

    lines.append(
        "ACTUAL_EXECUTION_ACCOUNTING = "
        + (
            "PASS"
            if audit[
                "accounting_pass"
            ]
            else "FAIL"
        )
    )

    lines.append(
        "PAIRED_POLICY_SEED_AUDIT = "
        + (
            "PASS"
            if audit[
                "paired_policy_seed_pass"
            ]
            else "FAIL"
        )
    )

    lines.append(
        "BSOP_GEOMETRIC_CONSTRAINTS = "
        + (
            "PASS"
            if constraint_pass
            else "FAIL"
        )
    )

    path.write_text(
        "\n".join(lines)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=20260811,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.run == REFERENCE_RUN:
        raise RuntimeError(
            "Refusing to overwrite the frozen "
            "seed42 analysis directory."
        )

    print(
        "===== SEED42 DETERMINISTIC REGRESSION ====="
    )

    (
        ref_df,
        _,
        ref_bsop_traces,
        _,
    ) = analyze_run_per_seed(
        REFERENCE_RUN
    )

    regression_check_per_seed(
        ref_df
    )

    ref_contrasts = build_contrasts(
        ref_df,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    regression_check_contrast_points(
        ref_contrasts
    )

    ref_binary = build_binary(
        ref_df,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    regression_check_binary_points(
        ref_binary
    )

    ref_mechanism = build_mechanism(
        ref_df,
        ref_bsop_traces,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    regression_check_mechanism_points(
        ref_mechanism
    )

    print()
    print(
        "PUSHT_IMAGE_SEED42_ANALYSIS_REGRESSION = PASS"
    )

    print()
    print(
        "===== TARGET ANALYSIS ====="
    )
    print(
        "run =",
        args.run,
    )

    (
        df,
        _,
        bsop_traces,
        audit,
    ) = analyze_run_per_seed(
        args.run
    )

    if len(df) != 50:
        raise RuntimeError(
            f"Expected 50 paired seeds, got {len(df)}"
        )

    contrasts = build_contrasts(
        df,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    binary = build_binary(
        df,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    mechanism = build_mechanism(
        df,
        bsop_traces,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    out = (
        TABLE_ROOT
        / args.run
    )

    if out.exists():
        existing = list(
            out.iterdir()
        )

        if existing and not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {out}\n"
                "Use --overwrite only if intentional."
            )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed_path = (
        out
        / "pusht_image_per_seed_metrics.csv"
    )

    contrast_path = (
        out
        / "pusht_image_paired_contrasts.csv"
    )

    binary_path = (
        out
        / "pusht_image_binary_outcomes.csv"
    )

    mechanism_path = (
        out
        / "pusht_image_bsop_mechanism.csv"
    )

    summary_path = (
        out
        / "pusht_image_summary.txt"
    )

    df.to_csv(
        per_seed_path,
        index=False,
    )

    contrasts.to_csv(
        contrast_path,
        index=False,
    )

    binary.to_csv(
        binary_path,
        index=False,
    )

    mechanism.to_csv(
        mechanism_path,
        index=False,
    )

    write_summary(
        summary_path,
        args.run,
        df,
        contrasts,
        binary,
        mechanism,
        audit,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    print()
    print(
        summary_path.read_text()
    )

    manifest_path = (
        out
        / "analysis_sha256.txt"
    )

    artifacts = [
        per_seed_path,
        contrast_path,
        binary_path,
        mechanism_path,
        summary_path,
    ]

    lines = []

    for path in artifacts:
        lines.append(
            f"{sha256_file(path)}  {path}"
        )

    manifest_path.write_text(
        "\n".join(lines)
        + "\n"
    )

    print(
        "===== ANALYSIS OUTPUT HASHES ====="
    )

    print(
        manifest_path.read_text()
    )

    print(
        "PUSHT_IMAGE_BSOP_ANALYSIS = PASS"
    )


if __name__ == "__main__":
    main()
