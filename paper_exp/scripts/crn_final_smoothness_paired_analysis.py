#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


TABLE_DIR = Path("paper_exp/tables")

SMOOTHNESS_INPUT = (
    TABLE_DIR
    / "crn_final_smoothness_episode_metrics.csv"
)

TIMING_INPUT = (
    TABLE_DIR
    / "crn_final_timing_episode_means.csv"
)

OUTPUT_CSV = (
    TABLE_DIR
    / "crn_final_smoothness_paired_statistics.csv"
)

OUTPUT_JSON = (
    TABLE_DIR
    / "crn_final_smoothness_paired_statistics.json"
)

OUTPUT_TEXT = (
    TABLE_DIR
    / "crn_final_smoothness_paired_statistics.txt"
)

EXPECTED_SEEDS = list(range(4300000, 4300100))

N_BOOTSTRAP = 100000
BOOTSTRAP_SEED = 20260711
TIE_TOLERANCE = 1.0e-12


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def summarize(values):
    values = np.asarray(values, dtype=float)

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def bootstrap_reduction(
    original,
    aasf,
    seed,
):
    original = np.asarray(original, dtype=float)
    aasf = np.asarray(aasf, dtype=float)

    rng = np.random.default_rng(seed)

    mean_reductions = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )
    relative_reductions = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    n = original.size
    chunk_size = 5000

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            N_BOOTSTRAP,
        )

        indices = rng.integers(
            0,
            n,
            size=(stop - start, n),
        )

        original_samples = original[indices]
        aasf_samples = aasf[indices]

        original_means = np.mean(
            original_samples,
            axis=1,
        )
        aasf_means = np.mean(
            aasf_samples,
            axis=1,
        )

        mean_reductions[start:stop] = (
            original_means - aasf_means
        )

        relative_reductions[start:stop] = (
            100.0
            * (
                original_means
                - aasf_means
            )
            / original_means
        )

    absolute_ci = np.percentile(
        mean_reductions,
        [2.5, 97.5],
    )

    relative_ci = np.percentile(
        relative_reductions,
        [2.5, 97.5],
    )

    return (
        [float(x) for x in absolute_ci],
        [float(x) for x in relative_ci],
    )


def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    estimates = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    n = values.size
    chunk_size = 5000

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            N_BOOTSTRAP,
        )

        indices = rng.integers(
            0,
            n,
            size=(stop - start, n),
        )

        estimates[start:stop] = np.mean(
            values[indices],
            axis=1,
        )

    return [
        float(x)
        for x in np.percentile(
            estimates,
            [2.5, 97.5],
        )
    ]


def rank_biserial(reduction):
    reduction = np.asarray(
        reduction,
        dtype=float,
    )

    nonzero = reduction[
        np.abs(reduction)
        > TIE_TOLERANCE
    ]

    if nonzero.size == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(nonzero),
        method="average",
    )

    positive = float(
        np.sum(ranks[nonzero > 0])
    )
    negative = float(
        np.sum(ranks[nonzero < 0])
    )

    return (
        positive - negative
    ) / (
        positive + negative
    )


def hodges_lehmann(reduction):
    reduction = np.asarray(
        reduction,
        dtype=float,
    )

    walsh = (
        reduction[:, None]
        + reduction[None, :]
    ) / 2.0

    return float(
        np.median(
            walsh[
                np.triu_indices(
                    reduction.size
                )
            ]
        )
    )


def exact_sign_test(wins, losses):
    total = wins + losses

    if total == 0:
        return 1.0

    return float(
        stats.binomtest(
            min(wins, losses),
            total,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    count = p_values.size
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(
        count,
        dtype=float,
    )

    running_maximum = 0.0

    for index, p_value in enumerate(
        sorted_p
    ):
        adjusted = (
            count - index
        ) * p_value

        running_maximum = max(
            running_maximum,
            adjusted,
        )

        adjusted_sorted[index] = min(
            1.0,
            running_maximum,
        )

    adjusted = np.empty(
        count,
        dtype=float,
    )
    adjusted[order] = adjusted_sorted

    return adjusted


smoothness_rows = read_csv(
    SMOOTHNESS_INPUT
)

timing_rows = read_csv(
    TIMING_INPUT
)

smoothness_seeds = [
    int(row["seed"])
    for row in smoothness_rows
]

timing_seeds = [
    int(row["seed"])
    for row in timing_rows
]

if smoothness_seeds != EXPECTED_SEEDS:
    raise RuntimeError(
        "Smoothness CSV seed mismatch."
    )

if timing_seeds != EXPECTED_SEEDS:
    raise RuntimeError(
        "Timing CSV seed mismatch."
    )


metric_definitions = [
    (
        "D1",
        "original_d1",
        "aasf_d1",
    ),
    (
        "D2",
        "original_d2",
        "aasf_d2",
    ),
    (
        "D3",
        "original_d3",
        "aasf_d3",
    ),
]

results = []
raw_wilcoxon_p = []

for metric_index, (
    metric,
    original_field,
    aasf_field,
) in enumerate(metric_definitions):

    original = np.asarray(
        [
            float(row[original_field])
            for row in smoothness_rows
        ],
        dtype=float,
    )

    aasf = np.asarray(
        [
            float(row[aasf_field])
            for row in smoothness_rows
        ],
        dtype=float,
    )

    # Positive means AASF is smoother.
    reduction = original - aasf

    original_summary = summarize(original)
    aasf_summary = summarize(aasf)
    reduction_summary = summarize(reduction)

    wins = int(
        np.sum(
            reduction > TIE_TOLERANCE
        )
    )
    losses = int(
        np.sum(
            reduction < -TIE_TOLERANCE
        )
    )
    ties = int(
        reduction.size
        - wins
        - losses
    )

    try:
        wilcoxon = stats.wilcoxon(
            reduction,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        wilcoxon = stats.wilcoxon(
            reduction,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )

    absolute_ci, relative_ci = (
        bootstrap_reduction(
            original,
            aasf,
            BOOTSTRAP_SEED
            + metric_index,
        )
    )

    relative_reduction = (
        100.0
        * (
            original_summary["mean"]
            - aasf_summary["mean"]
        )
        / original_summary["mean"]
    )

    reduction_sd = float(
        np.std(reduction, ddof=1)
    )

    cohen_dz = (
        float(
            np.mean(reduction)
            / reduction_sd
        )
        if reduction_sd > 0
        else None
    )

    result = {
        "metric": metric,
        "positive_direction": (
            "Original minus AASF; "
            "positive means smoother AASF"
        ),

        "original_mean": (
            original_summary["mean"]
        ),
        "original_sd": (
            original_summary["sd"]
        ),
        "original_median": (
            original_summary["median"]
        ),

        "aasf_mean": aasf_summary["mean"],
        "aasf_sd": aasf_summary["sd"],
        "aasf_median": (
            aasf_summary["median"]
        ),

        "mean_absolute_reduction": (
            reduction_summary["mean"]
        ),
        "absolute_reduction_ci_low": (
            absolute_ci[0]
        ),
        "absolute_reduction_ci_high": (
            absolute_ci[1]
        ),

        "relative_mean_reduction_percent": (
            relative_reduction
        ),
        "relative_reduction_ci_low_percent": (
            relative_ci[0]
        ),
        "relative_reduction_ci_high_percent": (
            relative_ci[1]
        ),

        "hodges_lehmann_reduction": (
            hodges_lehmann(reduction)
        ),

        "aasf_smoother_wins": wins,
        "ties": ties,
        "aasf_rougher_losses": losses,

        "probability_aasf_smoother_excluding_ties": (
            wins / (wins + losses)
            if wins + losses > 0
            else 0.5
        ),

        "exact_sign_test_p": (
            exact_sign_test(
                wins,
                losses,
            )
        ),

        "wilcoxon_statistic": float(
            wilcoxon.statistic
        ),
        "wilcoxon_raw_p": float(
            wilcoxon.pvalue
        ),

        "rank_biserial_correlation": (
            rank_biserial(reduction)
        ),

        "cohen_dz": cohen_dz,
    }

    results.append(result)
    raw_wilcoxon_p.append(
        float(wilcoxon.pvalue)
    )


adjusted_p = holm_adjust(
    raw_wilcoxon_p
)

for result, p_value in zip(
    results,
    adjusted_p,
):
    result[
        "wilcoxon_holm_adjusted_p"
    ] = float(p_value)


# Net AASF post-processing overhead.
overhead = np.asarray(
    [
        float(
            row[
                "paired_post_difference_ms"
            ]
        )
        for row in timing_rows
    ],
    dtype=float,
)

original_policy = np.asarray(
    [
        float(row["original_policy_ms"])
        for row in timing_rows
    ],
    dtype=float,
)

overhead_summary = summarize(overhead)

overhead_ci = bootstrap_mean_ci(
    overhead,
    BOOTSTRAP_SEED + 100,
)

relative_overhead = (
    100.0
    * np.mean(overhead)
    / np.mean(original_policy)
)


# Paired bootstrap CI for relative overhead.
rng = np.random.default_rng(
    BOOTSTRAP_SEED + 101
)

relative_samples = np.empty(
    N_BOOTSTRAP,
    dtype=float,
)

n = overhead.size
chunk_size = 5000

for start in range(
    0,
    N_BOOTSTRAP,
    chunk_size,
):
    stop = min(
        start + chunk_size,
        N_BOOTSTRAP,
    )

    indices = rng.integers(
        0,
        n,
        size=(stop - start, n),
    )

    relative_samples[start:stop] = (
        100.0
        * np.mean(
            overhead[indices],
            axis=1,
        )
        / np.mean(
            original_policy[indices],
            axis=1,
        )
    )

relative_overhead_ci = [
    float(x)
    for x in np.percentile(
        relative_samples,
        [2.5, 97.5],
    )
]


with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys()),
    )
    writer.writeheader()
    writer.writerows(results)


combined = {
    "analysis": {
        "paired_episodes": 100,
        "seed_start": EXPECTED_SEEDS[0],
        "seed_end": EXPECTED_SEEDS[-1],
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "multiple_testing_correction": (
            "Holm correction across D1, D2, and D3"
        ),
        "input_sha256": {
            str(SMOOTHNESS_INPUT): (
                sha256(SMOOTHNESS_INPUT)
            ),
            str(TIMING_INPUT): (
                sha256(TIMING_INPUT)
            ),
        },
    },
    "smoothness": results,
    "postprocessing_overhead": {
        "mean_ms": overhead_summary["mean"],
        "sd_ms": overhead_summary["sd"],
        "bootstrap_95_ci_ms": overhead_ci,
        "relative_to_original_policy_percent": (
            relative_overhead
        ),
        "relative_bootstrap_95_ci_percent": (
            relative_overhead_ci
        ),
    },
}

OUTPUT_JSON.write_text(
    json.dumps(
        combined,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)


lines = [
    "==== CRN final paired smoothness analysis ====",
    "",
    "Positive reductions mean that AASF is smoother.",
    (
        "Wilcoxon p-values use Holm correction "
        "across D1, D2, and D3."
    ),
    "",
]

for result in results:
    lines.extend([
        f"{result['metric']}:",
        (
            "  Original: "
            f"{result['original_mean']:.8f} "
            f"± {result['original_sd']:.8f}"
        ),
        (
            "  AASF    : "
            f"{result['aasf_mean']:.8f} "
            f"± {result['aasf_sd']:.8f}"
        ),
        (
            "  Mean reduction: "
            f"{result['mean_absolute_reduction']:+.8f} "
            "with bootstrap 95% CI "
            f"[{result['absolute_reduction_ci_low']:+.8f}, "
            f"{result['absolute_reduction_ci_high']:+.8f}]"
        ),
        (
            "  Relative mean reduction: "
            f"{result['relative_mean_reduction_percent']:+.3f}% "
            "with bootstrap 95% CI "
            f"[{result['relative_reduction_ci_low_percent']:+.3f}%, "
            f"{result['relative_reduction_ci_high_percent']:+.3f}%]"
        ),
        (
            "  Smoother/tie/rougher: "
            f"{result['aasf_smoother_wins']}/"
            f"{result['ties']}/"
            f"{result['aasf_rougher_losses']}"
        ),
        (
            "  Wilcoxon raw p="
            f"{result['wilcoxon_raw_p']:.10g}, "
            "Holm-adjusted p="
            f"{result['wilcoxon_holm_adjusted_p']:.10g}"
        ),
        (
            "  Rank-biserial correlation="
            f"{result['rank_biserial_correlation']:+.6f}"
        ),
        (
            "  Cohen dz="
            f"{result['cohen_dz']:+.6f}"
        ),
        "",
    ])

lines.extend([
    "Net AASF post-processing overhead:",
    (
        f"  {overhead_summary['mean']:.6f} "
        f"± {overhead_summary['sd']:.6f} ms"
    ),
    (
        "  Bootstrap 95% CI: "
        f"[{overhead_ci[0]:.6f}, "
        f"{overhead_ci[1]:.6f}] ms"
    ),
    (
        "  Relative to Original policy inference: "
        f"{relative_overhead:.6f}% "
        "with bootstrap 95% CI "
        f"[{relative_overhead_ci[0]:.6f}%, "
        f"{relative_overhead_ci[1]:.6f}%]"
    ),
    "",
    "Saved files:",
    f"  {OUTPUT_CSV}",
    f"  {OUTPUT_JSON}",
    f"  {OUTPUT_TEXT}",
])

report = "\n".join(lines) + "\n"

OUTPUT_TEXT.write_text(
    report,
    encoding="utf-8",
)

print(report, end="")
