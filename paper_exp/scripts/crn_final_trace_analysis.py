#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORIGINAL_DIR = Path(
    "paper_artifacts/traces/crn_final_original_100"
)
AASF_DIR = Path(
    "paper_artifacts/traces/crn_final_aasf_100"
)

TABLE_DIR = Path("paper_exp/tables")
FIGURE_DIR = Path("paper_exp/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = list(range(4300000, 4300100))

TAU = 10.0
ALPHA_MIN = 0.45
ALPHA_MAX = 0.80
SUCCESS_THRESHOLD = 0.90
BOUNDARY_TOLERANCE = 1.0e-6

REQUIRED_KEYS = {
    "seed",
    "method",
    "score",
    "warmup_queries",
    "raw_action",
    "filtered_action",
    "d",
    "alpha",
    "chunk_starts",
    "policy_inference_ms",
    "postprocess_ms",
    "decision_loop_ms",
}


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)

        if missing:
            raise RuntimeError(
                f"{path}: missing keys {sorted(missing)}"
            )

        trace = {
            "path": str(path),
            "seed": int(data["seed"]),
            "method": str(data["method"]),
            "score": float(data["score"]),
            "warmup_queries": int(data["warmup_queries"]),
            "raw_action": np.asarray(
                data["raw_action"],
                dtype=float,
            ),
            "filtered_action": np.asarray(
                data["filtered_action"],
                dtype=float,
            ),
            "d": np.asarray(
                data["d"],
                dtype=float,
            ),
            "alpha": np.asarray(
                data["alpha"],
                dtype=float,
            ),
            "chunk_starts": np.asarray(
                data["chunk_starts"],
                dtype=int,
            ),
            "policy_inference_ms": np.asarray(
                data["policy_inference_ms"],
                dtype=float,
            ),
            "postprocess_ms": np.asarray(
                data["postprocess_ms"],
                dtype=float,
            ),
            "decision_loop_ms": np.asarray(
                data["decision_loop_ms"],
                dtype=float,
            ),
        }

    raw = trace["raw_action"]
    filtered = trace["filtered_action"]
    d = trace["d"]
    alpha = trace["alpha"]

    if raw.ndim != 2:
        raise RuntimeError(
            f"{path}: expected 2-D action array, "
            f"got shape {raw.shape}"
        )

    if raw.shape != filtered.shape:
        raise RuntimeError(
            f"{path}: raw/filtered action shape mismatch"
        )

    if d.shape != (raw.shape[0],):
        raise RuntimeError(
            f"{path}: invalid d shape {d.shape}"
        )

    if alpha.shape != (raw.shape[0],):
        raise RuntimeError(
            f"{path}: invalid alpha shape {alpha.shape}"
        )

    timing_lengths = {
        trace["policy_inference_ms"].size,
        trace["postprocess_ms"].size,
        trace["decision_loop_ms"].size,
    }

    if len(timing_lengths) != 1:
        raise RuntimeError(
            f"{path}: timing array lengths differ"
        )

    if not np.all(
        trace["decision_loop_ms"]
        >= trace["policy_inference_ms"] - 1.0e-9
    ):
        raise RuntimeError(
            f"{path}: decision-loop time is below "
            "policy-inference time"
        )

    return trace


def load_directory(directory):
    files = sorted(directory.glob("*.npz"))

    if len(files) != 100:
        raise RuntimeError(
            f"{directory}: expected 100 NPZ files, "
            f"found {len(files)}"
        )

    result = {}

    for path in files:
        trace = load_trace(path)
        seed = trace["seed"]

        if seed in result:
            raise RuntimeError(
                f"{directory}: duplicate seed {seed}"
            )

        result[seed] = trace

    if sorted(result) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{directory}: seed set does not equal "
            "4300000-4300099"
        )

    return result


def sample_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError("Cannot summarize an empty array.")

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(
            np.std(values, ddof=1)
            if values.size > 1
            else 0.0
        ),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "iqr": float(
            np.quantile(values, 0.75)
            - np.quantile(values, 0.25)
        ),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def post_warmup_episode_timing(trace):
    warmup = trace["warmup_queries"]
    length = trace["policy_inference_ms"].size

    if length <= warmup:
        raise RuntimeError(
            f"{trace['path']}: only {length} timing samples "
            f"for a warm-up of {warmup}"
        )

    selection = slice(warmup, None)

    return {
        "policy_ms": float(
            np.mean(
                trace["policy_inference_ms"][selection]
            )
        ),
        "post_ms": float(
            np.mean(
                trace["postprocess_ms"][selection]
            )
        ),
        "loop_ms": float(
            np.mean(
                trace["decision_loop_ms"][selection]
            )
        ),
        "retained_queries": int(length - warmup),
    }


def action_difference_metric(actions, order):
    actions = np.asarray(actions, dtype=float)

    if actions.shape[0] <= order:
        return float("nan")

    differences = np.diff(
        actions,
        n=order,
        axis=0,
    )

    magnitudes = np.linalg.norm(
        differences,
        axis=1,
    )

    return float(np.mean(magnitudes))


def local_action_difference(actions, order):
    actions = np.asarray(actions, dtype=float)

    if actions.shape[0] <= order:
        return np.asarray([], dtype=float)

    differences = np.diff(
        actions,
        n=order,
        axis=0,
    )

    return np.linalg.norm(
        differences,
        axis=1,
    )


original = load_directory(ORIGINAL_DIR)
aasf = load_directory(AASF_DIR)

if sorted(original) != sorted(aasf):
    raise RuntimeError(
        "Original and AASF seed sets do not match."
    )


# =========================================================
# 1. Timing analysis
# =========================================================
timing_rows = []

for seed in EXPECTED_SEEDS:
    original_timing = post_warmup_episode_timing(
        original[seed]
    )
    aasf_timing = post_warmup_episode_timing(
        aasf[seed]
    )

    timing_rows.append({
        "seed": seed,

        "original_policy_ms": (
            original_timing["policy_ms"]
        ),
        "original_post_ms": (
            original_timing["post_ms"]
        ),
        "original_loop_ms": (
            original_timing["loop_ms"]
        ),
        "original_retained_queries": (
            original_timing["retained_queries"]
        ),

        "aasf_policy_ms": (
            aasf_timing["policy_ms"]
        ),
        "aasf_post_ms": (
            aasf_timing["post_ms"]
        ),
        "aasf_loop_ms": (
            aasf_timing["loop_ms"]
        ),
        "aasf_retained_queries": (
            aasf_timing["retained_queries"]
        ),

        "paired_policy_difference_ms": (
            aasf_timing["policy_ms"]
            - original_timing["policy_ms"]
        ),
        "paired_post_difference_ms": (
            aasf_timing["post_ms"]
            - original_timing["post_ms"]
        ),
        "paired_loop_difference_ms": (
            aasf_timing["loop_ms"]
            - original_timing["loop_ms"]
        ),
    })


TIMING_EPISODE_CSV = (
    TABLE_DIR
    / "crn_final_timing_episode_means.csv"
)

with TIMING_EPISODE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(timing_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(timing_rows)


def timing_column(name):
    return np.asarray(
        [row[name] for row in timing_rows],
        dtype=float,
    )


timing_summary_rows = []

for metric, original_field, aasf_field, diff_field in [
    (
        "policy_inference",
        "original_policy_ms",
        "aasf_policy_ms",
        "paired_policy_difference_ms",
    ),
    (
        "postprocessing",
        "original_post_ms",
        "aasf_post_ms",
        "paired_post_difference_ms",
    ),
    (
        "decision_loop",
        "original_loop_ms",
        "aasf_loop_ms",
        "paired_loop_difference_ms",
    ),
]:
    original_values = timing_column(
        original_field
    )
    aasf_values = timing_column(
        aasf_field
    )
    differences = timing_column(
        diff_field
    )

    original_summary = sample_summary(
        original_values
    )
    aasf_summary = sample_summary(
        aasf_values
    )
    difference_summary = sample_summary(
        differences
    )

    timing_summary_rows.append({
        "metric": metric,
        "unit": "ms",
        "aggregation": (
            "mean per episode after removing "
            "the recorded warm-up queries"
        ),
        "original_mean": original_summary["mean"],
        "original_sd": original_summary["sd"],
        "original_median": (
            original_summary["median"]
        ),
        "aasf_mean": aasf_summary["mean"],
        "aasf_sd": aasf_summary["sd"],
        "aasf_median": aasf_summary["median"],
        "paired_difference_mean": (
            difference_summary["mean"]
        ),
        "paired_difference_sd": (
            difference_summary["sd"]
        ),
        "paired_difference_median": (
            difference_summary["median"]
        ),
    })


TIMING_SUMMARY_CSV = (
    TABLE_DIR
    / "crn_final_timing_summary.csv"
)

with TIMING_SUMMARY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            timing_summary_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(timing_summary_rows)


original_policy_summary = sample_summary(
    timing_column("original_policy_ms")
)
original_post_summary = sample_summary(
    timing_column("original_post_ms")
)
original_loop_summary = sample_summary(
    timing_column("original_loop_ms")
)

aasf_policy_summary = sample_summary(
    timing_column("aasf_policy_ms")
)
aasf_post_summary = sample_summary(
    timing_column("aasf_post_ms")
)
aasf_loop_summary = sample_summary(
    timing_column("aasf_loop_ms")
)

net_overhead_summary = sample_summary(
    timing_column(
        "paired_post_difference_ms"
    )
)

relative_overhead_percent = (
    100.0
    * net_overhead_summary["mean"]
    / original_policy_summary["mean"]
)


# =========================================================
# 2. d_i and alpha_i analysis
# =========================================================
all_d = []
all_alpha = []
diagnostic_rows = []

for seed in EXPECTED_SEEDS:
    trace = aasf[seed]

    finite_d = trace["d"][
        np.isfinite(trace["d"])
    ]
    finite_alpha = trace["alpha"][
        np.isfinite(trace["alpha"])
    ]

    if finite_d.size == 0:
        raise RuntimeError(
            f"AASF seed {seed}: no finite d values"
        )

    if finite_alpha.size == 0:
        raise RuntimeError(
            f"AASF seed {seed}: no finite alpha values"
        )

    all_d.append(finite_d)
    all_alpha.append(finite_alpha)

    d_summary = sample_summary(finite_d)
    alpha_summary = sample_summary(
        finite_alpha
    )

    diagnostic_rows.append({
        "seed": seed,
        "score": trace["score"],
        "action_steps": int(
            trace["filtered_action"].shape[0]
        ),
        "policy_queries": int(
            trace["policy_inference_ms"].size
        ),
        "finite_d_count": int(finite_d.size),
        "mean_d": d_summary["mean"],
        "median_d": d_summary["median"],
        "p90_d": d_summary["p90"],
        "p95_d": d_summary["p95"],
        "fraction_d_gt_tau": float(
            np.mean(finite_d > TAU)
        ),
        "finite_alpha_count": int(
            finite_alpha.size
        ),
        "mean_alpha": alpha_summary["mean"],
        "sd_alpha": alpha_summary["sd"],
        "fraction_alpha_at_min": float(
            np.mean(
                np.isclose(
                    finite_alpha,
                    ALPHA_MIN,
                    atol=BOUNDARY_TOLERANCE,
                    rtol=0.0,
                )
            )
        ),
        "fraction_alpha_at_max": float(
            np.mean(
                np.isclose(
                    finite_alpha,
                    ALPHA_MAX,
                    atol=BOUNDARY_TOLERANCE,
                    rtol=0.0,
                )
            )
        ),
        "fraction_alpha_below_max": float(
            np.mean(
                finite_alpha
                < ALPHA_MAX
                - BOUNDARY_TOLERANCE
            )
        ),
    })


all_d = np.concatenate(all_d)
all_alpha = np.concatenate(all_alpha)

d_summary = sample_summary(all_d)
alpha_summary = sample_summary(all_alpha)

d_fraction_gt_tau = float(
    np.mean(all_d > TAU)
)

alpha_fraction_at_min = float(
    np.mean(
        np.isclose(
            all_alpha,
            ALPHA_MIN,
            atol=BOUNDARY_TOLERANCE,
            rtol=0.0,
        )
    )
)

alpha_fraction_at_max = float(
    np.mean(
        np.isclose(
            all_alpha,
            ALPHA_MAX,
            atol=BOUNDARY_TOLERANCE,
            rtol=0.0,
        )
    )
)

alpha_fraction_below_max = float(
    np.mean(
        all_alpha
        < ALPHA_MAX - BOUNDARY_TOLERANCE
    )
)


DIAGNOSTIC_CSV = (
    TABLE_DIR
    / "crn_final_aasf_episode_diagnostics.csv"
)

with DIAGNOSTIC_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            diagnostic_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(diagnostic_rows)


d_alpha_summary_rows = [
    {
        "variable": "d_i",
        "statistic": "count",
        "value": d_summary["count"],
    },
    {
        "variable": "d_i",
        "statistic": "median",
        "value": d_summary["median"],
    },
    {
        "variable": "d_i",
        "statistic": "q1",
        "value": d_summary["q1"],
    },
    {
        "variable": "d_i",
        "statistic": "q3",
        "value": d_summary["q3"],
    },
    {
        "variable": "d_i",
        "statistic": "iqr",
        "value": d_summary["iqr"],
    },
    {
        "variable": "d_i",
        "statistic": "p90",
        "value": d_summary["p90"],
    },
    {
        "variable": "d_i",
        "statistic": "p95",
        "value": d_summary["p95"],
    },
    {
        "variable": "d_i",
        "statistic": "fraction_above_tau_10",
        "value": d_fraction_gt_tau,
    },
    {
        "variable": "alpha_i",
        "statistic": "count",
        "value": alpha_summary["count"],
    },
    {
        "variable": "alpha_i",
        "statistic": "mean",
        "value": alpha_summary["mean"],
    },
    {
        "variable": "alpha_i",
        "statistic": "sd",
        "value": alpha_summary["sd"],
    },
    {
        "variable": "alpha_i",
        "statistic": "median",
        "value": alpha_summary["median"],
    },
    {
        "variable": "alpha_i",
        "statistic": "fraction_at_alpha_min_0.45",
        "value": alpha_fraction_at_min,
    },
    {
        "variable": "alpha_i",
        "statistic": "fraction_at_alpha_max_0.80",
        "value": alpha_fraction_at_max,
    },
    {
        "variable": "alpha_i",
        "statistic": "fraction_below_alpha_max",
        "value": alpha_fraction_below_max,
    },
]


D_ALPHA_SUMMARY_CSV = (
    TABLE_DIR
    / "crn_final_d_alpha_summary.csv"
)

with D_ALPHA_SUMMARY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "variable",
            "statistic",
            "value",
        ],
    )
    writer.writeheader()
    writer.writerows(d_alpha_summary_rows)


# Distribution figure.
distribution_png = (
    FIGURE_DIR
    / "crn_final_d_alpha_distributions.png"
)
distribution_pdf = (
    FIGURE_DIR
    / "crn_final_d_alpha_distributions.pdf"
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.3),
)

axes[0].hist(
    all_d,
    bins=60,
)
axes[0].axvline(
    TAU,
    linestyle="--",
    linewidth=1.5,
    label=r"$\tau=10$",
)
axes[0].set_xlabel(r"Action difference $d_i$")
axes[0].set_ylabel("Count")
axes[0].set_title(
    "Distribution of action differences"
)
axes[0].legend()

axes[1].hist(
    all_alpha,
    bins=50,
)
axes[1].axvline(
    ALPHA_MIN,
    linestyle="--",
    linewidth=1.2,
    label=r"$\alpha_{\min}=0.45$",
)
axes[1].axvline(
    ALPHA_MAX,
    linestyle="--",
    linewidth=1.2,
    label=r"$\alpha_{\max}=0.80$",
)
axes[1].set_xlabel(
    r"Adaptive coefficient $\alpha_i$"
)
axes[1].set_ylabel("Count")
axes[1].set_title(
    "Distribution of adaptive coefficients"
)
axes[1].legend()

figure.tight_layout()
figure.savefig(
    distribution_png,
    dpi=300,
    bbox_inches="tight",
)
figure.savefig(
    distribution_pdf,
    bbox_inches="tight",
)
plt.close(figure)


# =========================================================
# 3. Smoothness metrics and representative episode
# =========================================================
smoothness_rows = []

for seed in EXPECTED_SEEDS:
    original_actions = original[seed][
        "filtered_action"
    ]
    aasf_actions = aasf[seed][
        "filtered_action"
    ]

    original_d1 = action_difference_metric(
        original_actions,
        1,
    )
    original_d2 = action_difference_metric(
        original_actions,
        2,
    )
    original_d3 = action_difference_metric(
        original_actions,
        3,
    )

    aasf_d1 = action_difference_metric(
        aasf_actions,
        1,
    )
    aasf_d2 = action_difference_metric(
        aasf_actions,
        2,
    )
    aasf_d3 = action_difference_metric(
        aasf_actions,
        3,
    )

    smoothness_rows.append({
        "seed": seed,
        "original_score": original[seed]["score"],
        "aasf_score": aasf[seed]["score"],
        "aasf_minus_original_score": (
            aasf[seed]["score"]
            - original[seed]["score"]
        ),
        "original_action_steps": int(
            original_actions.shape[0]
        ),
        "aasf_action_steps": int(
            aasf_actions.shape[0]
        ),

        "original_d1": original_d1,
        "aasf_d1": aasf_d1,
        "aasf_minus_original_d1": (
            aasf_d1 - original_d1
        ),

        "original_d2": original_d2,
        "aasf_d2": aasf_d2,
        "aasf_minus_original_d2": (
            aasf_d2 - original_d2
        ),

        "original_d3": original_d3,
        "aasf_d3": aasf_d3,
        "aasf_minus_original_d3": (
            aasf_d3 - original_d3
        ),
        "d3_reduction_original_minus_aasf": (
            original_d3 - aasf_d3
        ),
    })


SMOOTHNESS_CSV = (
    TABLE_DIR
    / "crn_final_smoothness_episode_metrics.csv"
)

with SMOOTHNESS_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            smoothness_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(smoothness_rows)


valid_rows = [
    row
    for row in smoothness_rows
    if (
        row["original_score"]
        >= SUCCESS_THRESHOLD
        and row["aasf_score"]
        >= SUCCESS_THRESHOLD
        and np.isfinite(
            row[
                "d3_reduction_original_minus_aasf"
            ]
        )
    )
]

if not valid_rows:
    raise RuntimeError(
        "No episodes succeeded under both methods."
    )

selection_cutoff = None
candidates = None

for cutoff in [0.02, 0.05, 0.10, float("inf")]:
    selected = [
        row
        for row in valid_rows
        if (
            abs(
                row[
                    "aasf_minus_original_score"
                ]
            )
            <= cutoff
        )
    ]

    if len(selected) >= 5 or np.isinf(cutoff):
        candidates = selected
        selection_cutoff = cutoff
        break

if not candidates:
    raise RuntimeError(
        "Representative-episode candidate set is empty."
    )

candidate_reductions = np.asarray(
    [
        row[
            "d3_reduction_original_minus_aasf"
        ]
        for row in candidates
    ],
    dtype=float,
)

median_candidate_reduction = float(
    np.median(candidate_reductions)
)

representative = min(
    candidates,
    key=lambda row: (
        abs(
            row[
                "d3_reduction_original_minus_aasf"
            ]
            - median_candidate_reduction
        ),
        abs(
            row[
                "aasf_minus_original_score"
            ]
        ),
        row["seed"],
    ),
)

representative_seed = int(
    representative["seed"]
)

original_trace = original[
    representative_seed
]
aasf_trace = aasf[
    representative_seed
]

original_actions = original_trace[
    "filtered_action"
]
aasf_raw_actions = aasf_trace[
    "raw_action"
]
aasf_filtered_actions = aasf_trace[
    "filtered_action"
]

common_length = min(
    original_actions.shape[0],
    aasf_raw_actions.shape[0],
    aasf_filtered_actions.shape[0],
)

original_actions = original_actions[
    :common_length
]
aasf_raw_actions = aasf_raw_actions[
    :common_length
]
aasf_filtered_actions = aasf_filtered_actions[
    :common_length
]

action_steps = np.arange(common_length)

original_local_d3 = local_action_difference(
    original_actions,
    3,
)
aasf_raw_local_d3 = local_action_difference(
    aasf_raw_actions,
    3,
)
aasf_filtered_local_d3 = local_action_difference(
    aasf_filtered_actions,
    3,
)

d3_steps = np.arange(
    3,
    common_length,
)

alpha_series = aasf_trace["alpha"][
    :common_length
]

chunk_starts = aasf_trace[
    "chunk_starts"
]
chunk_starts = chunk_starts[
    chunk_starts < common_length
]


timeseries_png = (
    FIGURE_DIR
    / "crn_final_representative_action_timeseries.png"
)
timeseries_pdf = (
    FIGURE_DIR
    / "crn_final_representative_action_timeseries.pdf"
)

figure, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10.5),
    sharex=True,
)

for dimension in range(2):
    axes[dimension].plot(
        action_steps,
        original_actions[:, dimension],
        label="Original executed",
        linewidth=1.2,
    )
    axes[dimension].plot(
        action_steps,
        aasf_raw_actions[:, dimension],
        label="AASF raw",
        linewidth=1.0,
        linestyle=":",
    )
    axes[dimension].plot(
        action_steps,
        aasf_filtered_actions[:, dimension],
        label="AASF filtered",
        linewidth=1.2,
    )

    axes[dimension].set_ylabel(
        f"Action {dimension + 1}"
    )
    axes[dimension].grid(
        alpha=0.25
    )

    for boundary in chunk_starts:
        axes[dimension].axvline(
            boundary,
            linewidth=0.5,
            alpha=0.12,
        )

axes[0].legend(
    ncol=3,
    loc="upper right",
)

axes[2].plot(
    d3_steps,
    original_local_d3,
    label="Original executed",
    linewidth=1.1,
)
axes[2].plot(
    d3_steps,
    aasf_raw_local_d3,
    label="AASF raw",
    linewidth=1.0,
    linestyle=":",
)
axes[2].plot(
    d3_steps,
    aasf_filtered_local_d3,
    label="AASF filtered",
    linewidth=1.1,
)
axes[2].set_ylabel(
    "Local third-difference\nmagnitude"
)
axes[2].grid(alpha=0.25)
axes[2].legend(
    ncol=3,
    loc="upper right",
)

axes[3].plot(
    action_steps,
    alpha_series,
    label=r"$\alpha_i$",
    linewidth=1.1,
)
axes[3].axhline(
    ALPHA_MIN,
    linestyle="--",
    linewidth=1.0,
    label=r"$\alpha_{\min}$",
)
axes[3].axhline(
    ALPHA_MAX,
    linestyle="--",
    linewidth=1.0,
    label=r"$\alpha_{\max}$",
)

for boundary in chunk_starts:
    axes[3].axvline(
        boundary,
        linewidth=0.5,
        alpha=0.12,
    )

axes[3].set_ylabel(
    r"Adaptive coefficient $\alpha_i$"
)
axes[3].set_xlabel("Action step")
axes[3].grid(alpha=0.25)
axes[3].legend(
    ncol=3,
    loc="lower right",
)

figure.suptitle(
    "Representative CRN-paired episode "
    f"(seed {representative_seed})\n"
    f"Original score={original_trace['score']:.4f}, "
    f"AASF score={aasf_trace['score']:.4f}",
    fontsize=12,
)

figure.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.95]
)
figure.savefig(
    timeseries_png,
    dpi=300,
    bbox_inches="tight",
)
figure.savefig(
    timeseries_pdf,
    bbox_inches="tight",
)
plt.close(figure)


REPRESENTATIVE_TXT = (
    TABLE_DIR
    / "crn_final_representative_episode.txt"
)

cutoff_text = (
    "unrestricted"
    if np.isinf(selection_cutoff)
    else f"{selection_cutoff:.2f}"
)

representative_report = (
    "Representative-episode selection\n"
    "================================\n"
    "Selection rule:\n"
    "  1. Both Original and AASF scores must be >= 0.90.\n"
    "  2. Use the first score-difference cutoff among "
    "0.02, 0.05, 0.10, or unrestricted that yields "
    "at least five candidates.\n"
    "  3. Select the candidate whose D3 reduction is "
    "closest to the candidate-set median.\n\n"
    f"Applied absolute score-difference cutoff: "
    f"{cutoff_text}\n"
    f"Candidate episodes: {len(candidates)}\n"
    f"Median candidate D3 reduction: "
    f"{median_candidate_reduction:+.8f}\n"
    f"Selected seed: {representative_seed}\n"
    f"Original score: "
    f"{original_trace['score']:.8f}\n"
    f"AASF score: {aasf_trace['score']:.8f}\n"
    f"AASF - Original score: "
    f"{representative['aasf_minus_original_score']:+.8f}\n"
    f"Original D3: "
    f"{representative['original_d3']:.8f}\n"
    f"AASF D3: "
    f"{representative['aasf_d3']:.8f}\n"
    f"Original - AASF D3 reduction: "
    f"{representative['d3_reduction_original_minus_aasf']:+.8f}\n"
    f"Common plotted action steps: {common_length}\n"
)

REPRESENTATIVE_TXT.write_text(
    representative_report,
    encoding="utf-8",
)


# =========================================================
# 4. Combined JSON and text report
# =========================================================
combined_results = {
    "inputs": {
        "original_trace_directory": str(
            ORIGINAL_DIR
        ),
        "aasf_trace_directory": str(
            AASF_DIR
        ),
        "original_trace_count": len(original),
        "aasf_trace_count": len(aasf),
        "seed_start": EXPECTED_SEEDS[0],
        "seed_end": EXPECTED_SEEDS[-1],
    },
    "timing": {
        "aggregation": (
            "mean per episode after removal of "
            "the recorded warm-up queries"
        ),
        "original_policy_ms": (
            original_policy_summary
        ),
        "original_postprocess_ms": (
            original_post_summary
        ),
        "original_decision_loop_ms": (
            original_loop_summary
        ),
        "aasf_policy_ms": (
            aasf_policy_summary
        ),
        "aasf_postprocess_ms": (
            aasf_post_summary
        ),
        "aasf_decision_loop_ms": (
            aasf_loop_summary
        ),
        "net_postprocess_overhead_ms": (
            net_overhead_summary
        ),
        (
            "net_overhead_relative_to_"
            "original_policy_percent"
        ): relative_overhead_percent,
    },
    "adaptation": {
        "tau": TAU,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "d_i": d_summary,
        "fraction_d_above_tau": (
            d_fraction_gt_tau
        ),
        "alpha_i": alpha_summary,
        "fraction_alpha_at_min": (
            alpha_fraction_at_min
        ),
        "fraction_alpha_at_max": (
            alpha_fraction_at_max
        ),
        "fraction_alpha_below_max": (
            alpha_fraction_below_max
        ),
    },
    "representative_episode": {
        "seed": representative_seed,
        "selection_score_difference_cutoff": (
            None
            if np.isinf(selection_cutoff)
            else selection_cutoff
        ),
        "candidate_count": len(candidates),
        "original_score": (
            original_trace["score"]
        ),
        "aasf_score": aasf_trace["score"],
        "original_d3": (
            representative["original_d3"]
        ),
        "aasf_d3": representative["aasf_d3"],
        "d3_reduction_original_minus_aasf": (
            representative[
                "d3_reduction_original_minus_aasf"
            ]
        ),
        "common_plotted_action_steps": (
            common_length
        ),
    },
}


JSON_OUTPUT = (
    TABLE_DIR
    / "crn_final_trace_analysis.json"
)

JSON_OUTPUT.write_text(
    json.dumps(
        combined_results,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)


TEXT_OUTPUT = (
    TABLE_DIR
    / "crn_final_trace_analysis.txt"
)

report = f"""==== CRN final trace analysis ====

Input validation:
  Original traces: {len(original)}
  AASF traces    : {len(aasf)}
  Paired seeds   : {EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}

Timing after per-episode warm-up removal:
  Values are mean ± SD across 100 episode-level means.

  Original policy inference:
    {original_policy_summary['mean']:.6f} ± {original_policy_summary['sd']:.6f} ms

  Original post-processing/instrumentation:
    {original_post_summary['mean']:.6f} ± {original_post_summary['sd']:.6f} ms

  Original decision loop:
    {original_loop_summary['mean']:.6f} ± {original_loop_summary['sd']:.6f} ms

  AASF policy inference:
    {aasf_policy_summary['mean']:.6f} ± {aasf_policy_summary['sd']:.6f} ms

  AASF post-processing:
    {aasf_post_summary['mean']:.6f} ± {aasf_post_summary['sd']:.6f} ms

  AASF decision loop:
    {aasf_loop_summary['mean']:.6f} ± {aasf_loop_summary['sd']:.6f} ms

  Net AASF post-processing overhead:
    {net_overhead_summary['mean']:.6f} ± {net_overhead_summary['sd']:.6f} ms

  Net overhead relative to Original policy inference:
    {relative_overhead_percent:.6f}%

  Important:
    Differences in policy-inference or total-loop time
    must not be interpreted as acceleration caused by AASF.

Action-difference distribution:
  Finite d_i values: {d_summary['count']}
  Median            : {d_summary['median']:.6f}
  Q1                : {d_summary['q1']:.6f}
  Q3                : {d_summary['q3']:.6f}
  IQR               : {d_summary['iqr']:.6f}
  90th percentile   : {d_summary['p90']:.6f}
  95th percentile   : {d_summary['p95']:.6f}
  Fraction d_i > 10 : {d_fraction_gt_tau:.6%}

Adaptive-coefficient distribution:
  Finite alpha_i values : {alpha_summary['count']}
  Mean                  : {alpha_summary['mean']:.6f}
  SD                    : {alpha_summary['sd']:.6f}
  Median                : {alpha_summary['median']:.6f}
  At alpha_min = 0.45   : {alpha_fraction_at_min:.6%}
  At alpha_max = 0.80   : {alpha_fraction_at_max:.6%}
  Below alpha_max       : {alpha_fraction_below_max:.6%}

Representative episode:
  Seed               : {representative_seed}
  Original score     : {original_trace['score']:.8f}
  AASF score         : {aasf_trace['score']:.8f}
  Original D3        : {representative['original_d3']:.8f}
  AASF D3            : {representative['aasf_d3']:.8f}
  D3 reduction       : {representative['d3_reduction_original_minus_aasf']:+.8f}
  Plotted steps      : {common_length}

Saved outputs:
  {TIMING_EPISODE_CSV}
  {TIMING_SUMMARY_CSV}
  {D_ALPHA_SUMMARY_CSV}
  {DIAGNOSTIC_CSV}
  {SMOOTHNESS_CSV}
  {REPRESENTATIVE_TXT}
  {JSON_OUTPUT}
  {TEXT_OUTPUT}
  {distribution_png}
  {distribution_pdf}
  {timeseries_png}
  {timeseries_pdf}
"""

TEXT_OUTPUT.write_text(
    report,
    encoding="utf-8",
)

print(report, end="")
