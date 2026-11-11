from pathlib import Path
import csv
import json
import math
import hashlib
import numpy as np


RUN = "blockpush_bsop_aasf_confirmatory_fresh50"

RESULT_ROOT = (
    Path("paper_exp/results")
    / RUN
)

OUTPUT_ROOT = (
    Path("paper_exp/tables")
    / RUN
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

METHODS = [
    "original",
    "aasf",
    "bsop",
    "bsop_aasf",
]

SEEDS = list(
    range(4400100, 4400150)
)

PAIRS = [
    ("original", "aasf"),
    ("original", "bsop"),
    ("aasf", "bsop_aasf"),
    ("original", "bsop_aasf"),
]

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260808


def load_scores(method):
    path = (
        RESULT_ROOT
        / method
        / "eval_log.json"
    )

    with path.open() as f:
        data = json.load(f)

    scores = {}

    for seed in SEEDS:
        key = (
            f"test/sim_max_reward_{seed}"
        )

        if key not in data:
            raise RuntimeError(
                f"{method}: missing {key}"
            )

        scores[seed] = float(
            data[key]
        )

    return scores


def exact_two_sided_binomial(
    positive,
    negative,
):
    n = positive + negative

    if n == 0:
        return 1.0

    k = min(
        positive,
        negative,
    )

    probability = (
        2.0
        * sum(
            math.comb(n, i)
            for i in range(k + 1)
        )
        / (2 ** n)
    )

    return min(
        1.0,
        probability,
    )


def bootstrap_mean_ci(
    values,
    seed,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    samples = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    batch_size = 2000

    for start in range(
        0,
        BOOTSTRAP_SAMPLES,
        batch_size,
    ):
        n = min(
            batch_size,
            BOOTSTRAP_SAMPLES - start,
        )

        indices = rng.integers(
            0,
            len(x),
            size=(n, len(x)),
        )

        samples[
            start:start+n
        ] = x[indices].mean(
            axis=1
        )

    return (
        float(np.mean(x)),
        float(
            np.percentile(
                samples,
                2.5,
            )
        ),
        float(
            np.percentile(
                samples,
                97.5,
            )
        ),
    )


scores = {
    method: load_scores(method)
    for method in METHODS
}


PER_SEED_PATH = (
    OUTPUT_ROOT
    / "blockpush_binary_outcomes_per_seed.csv"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "blockpush_binary_outcomes.csv"
)

SUMMARY_TXT = (
    OUTPUT_ROOT
    / "blockpush_binary_outcomes.txt"
)


per_seed_rows = []

for seed in SEEDS:
    row = {
        "seed": seed,
    }

    for method in METHODS:
        score = scores[
            method
        ][seed]

        row[
            f"{method}_score"
        ] = score

        row[
            f"{method}_p1"
        ] = int(
            score > 0.4
        )

        row[
            f"{method}_p2"
        ] = int(
            score > 0.9
        )

    per_seed_rows.append(
        row
    )


with PER_SEED_PATH.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            per_seed_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        per_seed_rows
    )


summary_rows = []
lines = []

lines.append(
    "===== BLOCKPUSH PAIRED BINARY OUTCOMES ====="
)

for pair_index, (
    base,
    new,
) in enumerate(PAIRS):

    lines.append("")
    lines.append(
        f"--- {base} -> {new} ---"
    )

    for metric_index, (
        label,
        threshold,
    ) in enumerate([
        ("P1", 0.4),
        ("P2", 0.9),
    ]):

        base_binary = np.asarray(
            [
                scores[base][seed]
                > threshold
                for seed in SEEDS
            ],
            dtype=np.int64,
        )

        new_binary = np.asarray(
            [
                scores[new][seed]
                > threshold
                for seed in SEEDS
            ],
            dtype=np.int64,
        )

        base_success = int(
            base_binary.sum()
        )

        new_success = int(
            new_binary.sum()
        )

        gain = int(
            np.sum(
                (base_binary == 0)
                & (new_binary == 1)
            )
        )

        loss = int(
            np.sum(
                (base_binary == 1)
                & (new_binary == 0)
            )
        )

        same_success = int(
            np.sum(
                (base_binary == 1)
                & (new_binary == 1)
            )
        )

        same_failure = int(
            np.sum(
                (base_binary == 0)
                & (new_binary == 0)
            )
        )

        difference = (
            new_binary
            - base_binary
        ).astype(
            np.float64
        )

        mean_diff, low, high = (
            bootstrap_mean_ci(
                difference,
                BOOTSTRAP_SEED
                + pair_index * 10
                + metric_index,
            )
        )

        exact_p = (
            exact_two_sided_binomial(
                gain,
                loss,
            )
        )

        row = {
            "base": base,
            "new": new,
            "metric": label,
            "threshold": threshold,
            "base_success": base_success,
            "new_success": new_success,
            "gain_0_to_1": gain,
            "loss_1_to_0": loss,
            "same_success_1_to_1": (
                same_success
            ),
            "same_failure_0_to_0": (
                same_failure
            ),
            "paired_rate_difference": (
                mean_diff
            ),
            "ci95_low": low,
            "ci95_high": high,
            "exact_mcnemar_p": exact_p,
        }

        summary_rows.append(
            row
        )

        lines.append(
            f"{label}: "
            f"{base_success}/50 -> "
            f"{new_success}/50"
        )

        lines.append(
            "  paired transitions "
            f"gain/loss/same-success/"
            f"same-failure = "
            f"{gain}/{loss}/"
            f"{same_success}/"
            f"{same_failure}"
        )

        lines.append(
            "  paired rate difference = "
            f"{100*mean_diff:+.2f} pp, "
            "95% bootstrap CI = "
            f"[{100*low:+.2f}, "
            f"{100*high:+.2f}] pp"
        )

        lines.append(
            "  exact McNemar/binomial "
            f"p = {exact_p:.6g}"
        )


with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


SUMMARY_TXT.write_text(
    "\n".join(lines) + "\n"
)

print(
    SUMMARY_TXT.read_text()
)

print(
    "===== OUTPUT HASHES ====="
)

for path in [
    PER_SEED_PATH,
    SUMMARY_CSV,
    SUMMARY_TXT,
]:

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    print(
        digest,
        path,
    )
