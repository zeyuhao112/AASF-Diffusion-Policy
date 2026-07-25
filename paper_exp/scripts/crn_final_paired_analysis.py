import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


SEEDS = list(range(4300000, 4300100))

# Primary success criterion used in the previous analyses.
PRIMARY_THRESHOLD = 0.90

# The other thresholds are sensitivity analyses only.
SENSITIVITY_THRESHOLDS = [
    0.80,
    0.90,
    0.95,
    0.99,
]

TIE_TOLERANCE = 1e-12
N_BOOTSTRAP = 100000
BOOTSTRAP_SEED = 20260711

TABLE_DIR = Path("paper_exp/tables")

ORIGINAL_EVAL = Path(
    "paper_exp/results/"
    "crn_final_original_100/eval_log.json"
)

AASF_EVAL = Path(
    "paper_exp/results/"
    "crn_final_aasf_100/eval_log.json"
)

EXPECTED_HASHES = {
    ORIGINAL_EVAL: (
        "3cd3c8aa48de11e4c53d408a12eeedf4"
        "823110e724bc45e477d0af56348f1258"
    ),
    Path(
        "paper_exp/results/"
        "crn_final_original_100/run_metadata.txt"
    ): (
        "d1490022ba9dce21f638449c38075cf32"
        "f77c16648df6c7bb404414e53ccfca1"
    ),
    AASF_EVAL: (
        "209ba5c77a16f26039da44d90334811a"
        "7c352ba7410aa2d1a21b3d70ef0af921"
    ),
    Path(
        "paper_exp/results/"
        "crn_final_aasf_100/run_metadata.txt"
    ): (
        "84e1b1e6f5846235bb900254eb27d372"
        "7e656d9804d4fabbc68bab7c464697c2"
    ),
    Path(
        "paper_exp/tables/"
        "crn_final_integrity_audit.txt"
    ): (
        "45af1d0d6dfd5f25f167f775351c8463"
        "a0d103ac13e1e88bbecb56d2f1f325d6"
    ),
}

EPISODE_CSV = TABLE_DIR / (
    "crn_final_paired_episode_results.csv"
)

SENSITIVITY_CSV = TABLE_DIR / (
    "crn_final_threshold_sensitivity.csv"
)

JSON_OUTPUT = TABLE_DIR / (
    "crn_final_paired_statistics.json"
)

TEXT_OUTPUT = TABLE_DIR / (
    "crn_final_paired_statistics.txt"
)


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_frozen_inputs():
    for path, expected_hash in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        actual_hash = calculate_sha256(path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                "Hash mismatch for {}\n"
                "Expected: {}\n"
                "Actual  : {}".format(
                    path,
                    expected_hash,
                    actual_hash,
                )
            )


def load_eval_scores(path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    prefix = "test/sim_max_reward_"
    result = {}

    for key, value in data.items():
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix):]

        if suffix.isdigit():
            result[int(suffix)] = float(value)

    if sorted(result) != SEEDS:
        raise RuntimeError(
            "Expected seeds {}-{} in {}, "
            "but found {} score entries.".format(
                SEEDS[0],
                SEEDS[-1],
                path,
                len(result),
            )
        )

    return result


def summarize(values):
    values = np.asarray(values, dtype=float)

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_bootstrap_ci(
    values,
    statistic,
    seed,
):
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)
    estimates = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    sample_count = values.size
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
            sample_count,
            size=(
                stop - start,
                sample_count,
            ),
        )

        samples = values[indices]

        if statistic == "mean":
            estimates[start:stop] = np.mean(
                samples,
                axis=1,
            )

        elif statistic == "median":
            estimates[start:stop] = np.median(
                samples,
                axis=1,
            )

        else:
            raise ValueError(statistic)

    lower, upper = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def exact_binomial_two_sided(
    smaller_count,
    total_count,
):
    if total_count == 0:
        return 1.0

    try:
        result = stats.binomtest(
            smaller_count,
            total_count,
            p=0.5,
            alternative="two-sided",
        )

        return float(result.pvalue)

    except AttributeError:
        return float(
            stats.binom_test(
                smaller_count,
                total_count,
                p=0.5,
                alternative="two-sided",
            )
        )


def calculate_mcnemar(
    original_success,
    aasf_success,
):
    both_success = int(
        np.sum(
            original_success
            & aasf_success
        )
    )

    original_only = int(
        np.sum(
            original_success
            & ~aasf_success
        )
    )

    aasf_only = int(
        np.sum(
            ~original_success
            & aasf_success
        )
    )

    both_fail = int(
        np.sum(
            ~original_success
            & ~aasf_success
        )
    )

    discordant = original_only + aasf_only

    p_value = exact_binomial_two_sided(
        min(
            original_only,
            aasf_only,
        ),
        discordant,
    )

    return {
        "both_success": both_success,
        "original_only_success": original_only,
        "aasf_only_success": aasf_only,
        "both_fail": both_fail,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p_value,
    }


def calculate_rank_biserial(diff):
    nonzero = diff[
        np.abs(diff) > TIE_TOLERANCE
    ]

    if nonzero.size == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(nonzero),
        method="average",
    )

    positive_rank_sum = float(
        np.sum(ranks[nonzero > 0])
    )

    negative_rank_sum = float(
        np.sum(ranks[nonzero < 0])
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    return (
        positive_rank_sum
        - negative_rank_sum
    ) / denominator


TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

verify_frozen_inputs()

original_map = load_eval_scores(
    ORIGINAL_EVAL
)

aasf_map = load_eval_scores(
    AASF_EVAL
)

original = np.asarray(
    [
        original_map[seed]
        for seed in SEEDS
    ],
    dtype=float,
)

aasf = np.asarray(
    [
        aasf_map[seed]
        for seed in SEEDS
    ],
    dtype=float,
)

# All paired quantities are defined as:
# AASF minus Original.
difference = aasf - original

wins = int(
    np.sum(
        difference > TIE_TOLERANCE
    )
)

losses = int(
    np.sum(
        difference < -TIE_TOLERANCE
    )
)

ties = int(
    difference.size
    - wins
    - losses
)

non_ties = wins + losses

mean_difference_ci = paired_bootstrap_ci(
    difference,
    statistic="mean",
    seed=BOOTSTRAP_SEED,
)

median_difference_ci = paired_bootstrap_ci(
    difference,
    statistic="median",
    seed=BOOTSTRAP_SEED + 1,
)

try:
    wilcoxon_result = stats.wilcoxon(
        difference,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )

except TypeError:
    # Compatibility with older SciPy.
    wilcoxon_result = stats.wilcoxon(
        difference,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
    )

rank_biserial = calculate_rank_biserial(
    difference
)

difference_sd = float(
    np.std(
        difference,
        ddof=1,
    )
)

if difference_sd > 0:
    cohen_dz = float(
        np.mean(difference)
        / difference_sd
    )
else:
    cohen_dz = None

# Hodges-Lehmann paired location-shift estimate.
walsh_averages = (
    difference[:, None]
    + difference[None, :]
) / 2.0

hodges_lehmann = float(
    np.median(
        walsh_averages[
            np.triu_indices(
                difference.size
            )
        ]
    )
)

sign_test_p = exact_binomial_two_sided(
    min(wins, losses),
    non_ties,
)

if non_ties > 0:
    probability_superiority = (
        wins / non_ties
    )
else:
    probability_superiority = 0.5


# Primary binary success analysis.
original_success = (
    original >= PRIMARY_THRESHOLD
)

aasf_success = (
    aasf >= PRIMARY_THRESHOLD
)

success_difference_per_pair = (
    aasf_success.astype(float)
    - original_success.astype(float)
)

success_rate_ci = paired_bootstrap_ci(
    success_difference_per_pair,
    statistic="mean",
    seed=BOOTSTRAP_SEED + 2,
)

primary_mcnemar = calculate_mcnemar(
    original_success,
    aasf_success,
)


# Save per-seed paired results.
with EPISODE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "seed",
        "original_score",
        "aasf_score",
        "aasf_minus_original",
        "original_success_at_0.90",
        "aasf_success_at_0.90",
        "paired_outcome",
    ])

    for (
        seed,
        original_score,
        aasf_score,
        delta,
    ) in zip(
        SEEDS,
        original,
        aasf,
        difference,
    ):
        if delta > TIE_TOLERANCE:
            outcome = "AASF_win"

        elif delta < -TIE_TOLERANCE:
            outcome = "AASF_loss"

        else:
            outcome = "tie"

        writer.writerow([
            seed,
            "{:.17g}".format(
                original_score
            ),
            "{:.17g}".format(
                aasf_score
            ),
            "{:.17g}".format(delta),
            int(
                original_score
                >= PRIMARY_THRESHOLD
            ),
            int(
                aasf_score
                >= PRIMARY_THRESHOLD
            ),
            outcome,
        ])


# Threshold sensitivity analysis.
sensitivity_rows = []

for threshold in SENSITIVITY_THRESHOLDS:
    original_threshold_success = (
        original >= threshold
    )

    aasf_threshold_success = (
        aasf >= threshold
    )

    row = {
        "threshold": threshold,
        "original_success_count": int(
            np.sum(
                original_threshold_success
            )
        ),
        "original_success_rate": float(
            np.mean(
                original_threshold_success
            )
        ),
        "aasf_success_count": int(
            np.sum(
                aasf_threshold_success
            )
        ),
        "aasf_success_rate": float(
            np.mean(
                aasf_threshold_success
            )
        ),
        "aasf_minus_original_rate": float(
            np.mean(
                aasf_threshold_success
            )
            - np.mean(
                original_threshold_success
            )
        ),
    }

    row.update(
        calculate_mcnemar(
            original_threshold_success,
            aasf_threshold_success,
        )
    )

    sensitivity_rows.append(row)


with SENSITIVITY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            sensitivity_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        sensitivity_rows
    )


original_stats = summarize(original)
aasf_stats = summarize(aasf)
difference_stats = summarize(
    difference
)

paired_results = dict(
    difference_stats
)

paired_results.update({
    "mean_bootstrap_95_ci": list(
        mean_difference_ci
    ),
    "median_bootstrap_95_ci": list(
        median_difference_ci
    ),
    "hodges_lehmann_shift": (
        hodges_lehmann
    ),
    "wins": wins,
    "ties": ties,
    "losses": losses,
    (
        "probability_aasf_better_"
        "excluding_ties"
    ): probability_superiority,
    "exact_sign_test_p": (
        sign_test_p
    ),
    "wilcoxon_statistic": float(
        wilcoxon_result.statistic
    ),
    "wilcoxon_two_sided_p": float(
        wilcoxon_result.pvalue
    ),
    "rank_biserial_correlation": (
        rank_biserial
    ),
    "cohen_dz": cohen_dz,
})

success_results = {
    "threshold": PRIMARY_THRESHOLD,
    "original_success_count": int(
        np.sum(original_success)
    ),
    "original_success_rate": float(
        np.mean(original_success)
    ),
    "aasf_success_count": int(
        np.sum(aasf_success)
    ),
    "aasf_success_rate": float(
        np.mean(aasf_success)
    ),
    "aasf_minus_original_rate": float(
        np.mean(aasf_success)
        - np.mean(original_success)
    ),
    "rate_difference_bootstrap_95_ci": (
        list(success_rate_ci)
    ),
}

success_results.update(
    primary_mcnemar
)

all_results = {
    "analysis": {
        "comparison": (
            "AASF minus Original"
        ),
        "paired_episodes": 100,
        "seed_range": [
            SEEDS[0],
            SEEDS[-1],
        ],
        "tie_tolerance": (
            TIE_TOLERANCE
        ),
        "bootstrap_resamples": (
            N_BOOTSTRAP
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "primary_success_threshold": (
            PRIMARY_THRESHOLD
        ),
        "python_version": (
            platform.python_version()
        ),
        "numpy_version": (
            np.__version__
        ),
        "scipy_version": (
            scipy.__version__
        ),
    },
    "original_score": (
        original_stats
    ),
    "aasf_score": aasf_stats,
    "paired_difference": (
        paired_results
    ),
    "primary_success_analysis": (
        success_results
    ),
    "input_sha256": {
        str(path): expected_hash
        for (
            path,
            expected_hash,
        ) in EXPECTED_HASHES.items()
    },
}

JSON_OUTPUT.write_text(
    json.dumps(
        all_results,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n",
    encoding="utf-8",
)


display_cohen_dz = (
    cohen_dz
    if cohen_dz is not None
    else float("nan")
)

report = """==== CRN final paired statistical analysis ====

Input verification:
  SHA-256 checks passed for all five frozen inputs.
  Paired seeds: {seed_start}-{seed_end} (n=100)

Score descriptives:
  Original: mean={original_mean:.8f}, SD={original_sd:.8f}, median={original_median:.8f}, IQR=[{original_q1:.8f}, {original_q3:.8f}]
  AASF    : mean={aasf_mean:.8f}, SD={aasf_sd:.8f}, median={aasf_median:.8f}, IQR=[{aasf_q1:.8f}, {aasf_q3:.8f}]

Paired score difference (AASF - Original):
  Mean difference={difference_mean:+.8f}, paired bootstrap 95% CI=[{mean_ci_low:+.8f}, {mean_ci_high:+.8f}]
  Median difference={difference_median:+.8f}, paired bootstrap 95% CI=[{median_ci_low:+.8f}, {median_ci_high:+.8f}]
  Hodges-Lehmann shift={hodges_lehmann:+.8f}
  Win/tie/loss={wins}/{ties}/{losses}
  P(AASF better | non-tie)={probability_superiority:.6f}
  Exact sign-test p={sign_test_p:.10g}
  Wilcoxon signed-rank W={wilcoxon_w:.6f}, two-sided p={wilcoxon_p:.10g}
  Rank-biserial correlation={rank_biserial:+.6f}
  Cohen dz={cohen_dz:+.6f}

Primary success analysis (score >= {threshold:.2f}):
  Original: {original_success_count}/100 ({original_success_rate:.1%})
  AASF    : {aasf_success_count}/100 ({aasf_success_rate:.1%})
  Rate difference={success_difference:+.1%}, paired bootstrap 95% CI=[{success_ci_low:+.1%}, {success_ci_high:+.1%}]
  Discordant pairs: Original-only={original_only}, AASF-only={aasf_only}
  Exact McNemar p={mcnemar_p:.10g}

Saved files:
  {episode_csv}
  {sensitivity_csv}
  {json_output}
  {text_output}
""".format(
    seed_start=SEEDS[0],
    seed_end=SEEDS[-1],

    original_mean=original_stats["mean"],
    original_sd=original_stats["sd"],
    original_median=(
        original_stats["median"]
    ),
    original_q1=original_stats["q1"],
    original_q3=original_stats["q3"],

    aasf_mean=aasf_stats["mean"],
    aasf_sd=aasf_stats["sd"],
    aasf_median=aasf_stats["median"],
    aasf_q1=aasf_stats["q1"],
    aasf_q3=aasf_stats["q3"],

    difference_mean=(
        difference_stats["mean"]
    ),
    mean_ci_low=mean_difference_ci[0],
    mean_ci_high=mean_difference_ci[1],

    difference_median=(
        difference_stats["median"]
    ),
    median_ci_low=(
        median_difference_ci[0]
    ),
    median_ci_high=(
        median_difference_ci[1]
    ),

    hodges_lehmann=hodges_lehmann,
    wins=wins,
    ties=ties,
    losses=losses,

    probability_superiority=(
        probability_superiority
    ),
    sign_test_p=sign_test_p,

    wilcoxon_w=float(
        wilcoxon_result.statistic
    ),
    wilcoxon_p=float(
        wilcoxon_result.pvalue
    ),

    rank_biserial=rank_biserial,
    cohen_dz=display_cohen_dz,

    threshold=PRIMARY_THRESHOLD,

    original_success_count=int(
        np.sum(original_success)
    ),
    original_success_rate=float(
        np.mean(original_success)
    ),

    aasf_success_count=int(
        np.sum(aasf_success)
    ),
    aasf_success_rate=float(
        np.mean(aasf_success)
    ),

    success_difference=float(
        np.mean(aasf_success)
        - np.mean(original_success)
    ),
    success_ci_low=success_rate_ci[0],
    success_ci_high=success_rate_ci[1],

    original_only=primary_mcnemar[
        "original_only_success"
    ],
    aasf_only=primary_mcnemar[
        "aasf_only_success"
    ],
    mcnemar_p=primary_mcnemar[
        "mcnemar_exact_p"
    ],

    episode_csv=EPISODE_CSV,
    sensitivity_csv=SENSITIVITY_CSV,
    json_output=JSON_OUTPUT,
    text_output=TEXT_OUTPUT,
)

TEXT_OUTPUT.write_text(
    report,
    encoding="utf-8",
)

print(report, end="")
