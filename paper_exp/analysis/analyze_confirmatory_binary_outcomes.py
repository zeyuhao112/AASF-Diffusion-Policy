from pathlib import Path
import csv
import hashlib
import math

import numpy as np


INPUT_PATH = Path(
    "paper_exp/tables/"
    "bsop_aasf_confirmatory_fresh50/"
    "confirmatory_paired_contrasts.csv"
)

OUTPUT_DIR = Path(
    "paper_exp/tables/"
    "bsop_aasf_confirmatory_fresh50"
)

OUTPUT_TEXT = (
    OUTPUT_DIR
    / "confirmatory_binary_outcomes.txt"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "confirmatory_binary_outcomes.csv"
)

EXPECTED_SEEDS = set(
    range(4300100, 4300150)
)

CONTRASTS = [
    "aasf_to_bsop_aasf",
    "original_to_bsop_aasf",
]

OUTCOMES = [
    (
        "SR@0.90",
        "old_sr090",
        "new_sr090",
    ),
    (
        "Exact success",
        "old_success",
        "new_success",
    ),
]

BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260806


def exact_mcnemar(old, new):
    old = np.asarray(
        old,
        dtype=np.int64,
    )
    new = np.asarray(
        new,
        dtype=np.int64,
    )

    if old.shape != new.shape:
        raise RuntimeError(
            "Old and new arrays have "
            "different shapes."
        )

    if not (
        np.all(np.isin(old, [0, 1]))
        and np.all(np.isin(new, [0, 1]))
    ):
        raise RuntimeError(
            "McNemar inputs must be binary."
        )

    both_fail = int(
        np.sum(
            (old == 0)
            & (new == 0)
        )
    )
    gained = int(
        np.sum(
            (old == 0)
            & (new == 1)
        )
    )
    lost = int(
        np.sum(
            (old == 1)
            & (new == 0)
        )
    )
    both_success = int(
        np.sum(
            (old == 1)
            & (new == 1)
        )
    )

    discordant = gained + lost

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(
            gained,
            lost,
        )

        lower_tail = sum(
            math.comb(
                discordant,
                index,
            )
            for index in range(
                smaller + 1
            )
        ) / (2.0 ** discordant)

        p_value = min(
            1.0,
            2.0 * lower_tail,
        )

    return {
        "both_fail": both_fail,
        "gained": gained,
        "lost": lost,
        "both_success": both_success,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def paired_risk_difference(
    old,
    new,
    seed,
):
    old = np.asarray(
        old,
        dtype=np.float64,
    )
    new = np.asarray(
        new,
        dtype=np.float64,
    )

    difference = new - old
    number_pairs = difference.size

    rng = np.random.default_rng(seed)

    bootstrap_indices = rng.integers(
        0,
        number_pairs,
        size=(
            BOOTSTRAP_SAMPLES,
            number_pairs,
        ),
    )

    bootstrap_means = np.mean(
        difference[
            bootstrap_indices
        ],
        axis=1,
    )

    return {
        "difference": float(
            np.mean(difference)
        ),
        "ci_low": float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        "ci_high": float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    }


if not INPUT_PATH.is_file():
    raise FileNotFoundError(
        f"Missing input file: {INPUT_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with INPUT_PATH.open(
    newline="",
) as file:
    input_rows = list(
        csv.DictReader(file)
    )


result_rows = []
lines = [
    "===== CONFIRMATORY PAIRED BINARY OUTCOMES =====",
    "Locked fresh seeds: 4300100-4300149",
    "Risk difference = new method minus old method.",
    (
        "Confidence intervals use a deterministic "
        "paired bootstrap with 100,000 samples."
    ),
    "",
]


for contrast_index, contrast in enumerate(
    CONTRASTS
):
    selected = [
        row
        for row in input_rows
        if row["contrast"] == contrast
    ]

    if len(selected) != 50:
        raise RuntimeError(
            f"{contrast}: expected 50 rows, "
            f"found {len(selected)}."
        )

    selected.sort(
        key=lambda row: int(
            row["seed"]
        )
    )

    seeds = {
        int(row["seed"])
        for row in selected
    }

    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{contrast}: seed set mismatch."
        )

    lines.extend([
        "=" * 72,
        contrast,
        "=" * 72,
    ])

    for outcome_index, (
        outcome_name,
        old_key,
        new_key,
    ) in enumerate(OUTCOMES):
        old = np.asarray(
            [
                int(row[old_key])
                for row in selected
            ],
            dtype=np.int64,
        )
        new = np.asarray(
            [
                int(row[new_key])
                for row in selected
            ],
            dtype=np.int64,
        )

        mcnemar = exact_mcnemar(
            old,
            new,
        )

        risk = paired_risk_difference(
            old,
            new,
            BOOTSTRAP_SEED
            + contrast_index * 100
            + outcome_index,
        )

        result = {
            "contrast": contrast,
            "outcome": outcome_name,
            "n_pairs": int(
                old.size
            ),
            "old_total": int(
                np.sum(old)
            ),
            "new_total": int(
                np.sum(new)
            ),
            "both_fail": (
                mcnemar["both_fail"]
            ),
            "gained": (
                mcnemar["gained"]
            ),
            "lost": (
                mcnemar["lost"]
            ),
            "both_success": (
                mcnemar["both_success"]
            ),
            "discordant_pairs": (
                mcnemar["discordant"]
            ),
            "paired_risk_difference": (
                risk["difference"]
            ),
            "bootstrap_ci_low": (
                risk["ci_low"]
            ),
            "bootstrap_ci_high": (
                risk["ci_high"]
            ),
            "exact_mcnemar_p": (
                mcnemar["p_value"]
            ),
        }

        result_rows.append(result)

        lines.extend([
            outcome_name,
            (
                f"  totals: "
                f"{result['old_total']}/50 "
                f"-> {result['new_total']}/50"
            ),
            (
                "  paired table "
                "(both fail / gained / lost / both succeed): "
                f"{result['both_fail']} / "
                f"{result['gained']} / "
                f"{result['lost']} / "
                f"{result['both_success']}"
            ),
            (
                "  paired risk difference: "
                f"{100.0 * result['paired_risk_difference']:+.2f} "
                "percentage points"
            ),
            (
                "  bootstrap 95% CI: "
                f"[{100.0 * result['bootstrap_ci_low']:+.2f}, "
                f"{100.0 * result['bootstrap_ci_high']:+.2f}] "
                "percentage points"
            ),
            (
                "  exact McNemar p: "
                f"{result['exact_mcnemar_p']:.6g}"
            ),
            "",
        ])


OUTPUT_TEXT.write_text(
    "\n".join(lines) + "\n"
)

with OUTPUT_CSV.open(
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            result_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(
        result_rows
    )


print(
    OUTPUT_TEXT.read_text(),
    end="",
)

print("===== OUTPUT SHA256 =====")

for path in [
    OUTPUT_TEXT,
    OUTPUT_CSV,
]:
    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    print(
        f"{digest}  {path}"
    )
