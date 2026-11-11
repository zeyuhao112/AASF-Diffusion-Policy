from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


OLD_PATH = Path(
    "paper_exp/tables/"
    "bsop_cross_task_master_v3/"
    "bsop_cross_task_modality_master_7cond.csv"
)

LOWDIM_PATH = Path(
    "paper_exp/tables/"
    "pusht_bsop_multicheckpoint_replication_v2/"
    "pusht_bsop_multicheckpoint_summary_v2.csv"
)

OUT = Path(
    "paper_exp/tables/"
    "bsop_cross_task_master_v4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUT
    / "bsop_cross_task_modality_master_7cond_v4.csv"
)

TXT_PATH = (
    OUT
    / "bsop_cross_task_modality_master_7cond_v4.txt"
)

AUDIT_PATH = (
    OUT
    / "v3_to_v4_provenance_audit.csv"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


master = pd.read_csv(
    OLD_PATH
)

lowdim = pd.read_csv(
    LOWDIM_PATH
)

expected_columns = {
    "task",
    "modality",
    "checkpoint",
    "primary",
    "primary_before",
    "primary_after",
    "secondary_before",
    "secondary_after",
    "dScore",
    "dScore_ci_low",
    "dScore_ci_high",
    "D1_pct",
    "D1_ci_low",
    "D1_ci_high",
    "D2_pct",
    "D2_ci_low",
    "D2_ci_high",
    "D3_pct",
    "D3_ci_low",
    "D3_ci_high",
    "Boundary_pct",
    "Boundary_ci_low",
    "Boundary_ci_high",
}

missing = (
    expected_columns
    - set(master.columns)
)

if missing:
    raise RuntimeError(
        "v3 master missing columns: "
        + repr(sorted(missing))
    )

if len(master) != 7:
    raise RuntimeError(
        f"expected 7 master rows, "
        f"got {len(master)}"
    )

audit_rows = []

for checkpoint in [
    "seed42",
    "seed43",
    "seed44",
]:

    src = lowdim[
        lowdim["checkpoint"]
        == checkpoint
    ]

    if len(src) != 1:
        raise RuntimeError(
            f"{checkpoint}: canonical "
            "lowdim row not unique"
        )

    src = src.iloc[0]

    mask = (
        (master["task"] == "Push-T")
        & (
            master["modality"]
            == "lowdim"
        )
        & (
            master["checkpoint"]
            == checkpoint
        )
    )

    if int(mask.sum()) != 1:
        raise RuntimeError(
            f"{checkpoint}: v3 master "
            "target row not unique"
        )

    idx = master.index[mask][0]

    # Success-count invariance audit.
    count_checks = {
        "primary_before": (
            int(master.at[
                idx,
                "primary_before"
            ]),
            int(src["sr090_before"]),
        ),
        "primary_after": (
            int(master.at[
                idx,
                "primary_after"
            ]),
            int(src["sr090_after"]),
        ),
        "secondary_before": (
            int(master.at[
                idx,
                "secondary_before"
            ]),
            int(src["exact_before"]),
        ),
        "secondary_after": (
            int(master.at[
                idx,
                "secondary_after"
            ]),
            int(src["exact_after"]),
        ),
    }

    for label, (
        old_value,
        new_value,
    ) in count_checks.items():
        if old_value != new_value:
            raise RuntimeError(
                f"{checkpoint}: "
                f"{label} changed "
                f"{old_value} -> "
                f"{new_value}"
            )

    replacement = {
        "dScore":
            src[
                "score_difference_mean"
            ],
        "dScore_ci_low":
            src[
                "score_difference_ci_low"
            ],
        "dScore_ci_high":
            src[
                "score_difference_ci_high"
            ],

        "D1_pct":
            src[
                "common_D1_reduction_pct_mean"
            ],
        "D1_ci_low":
            src[
                "common_D1_reduction_pct_ci_low"
            ],
        "D1_ci_high":
            src[
                "common_D1_reduction_pct_ci_high"
            ],

        "D2_pct":
            src[
                "common_D2_reduction_pct_mean"
            ],
        "D2_ci_low":
            src[
                "common_D2_reduction_pct_ci_low"
            ],
        "D2_ci_high":
            src[
                "common_D2_reduction_pct_ci_high"
            ],

        "D3_pct":
            src[
                "common_D3_reduction_pct_mean"
            ],
        "D3_ci_low":
            src[
                "common_D3_reduction_pct_ci_low"
            ],
        "D3_ci_high":
            src[
                "common_D3_reduction_pct_ci_high"
            ],

        "Boundary_pct":
            src[
                "boundary_D1_reduction_pct_mean"
            ],
        "Boundary_ci_low":
            src[
                "boundary_D1_reduction_pct_ci_low"
            ],
        "Boundary_ci_high":
            src[
                "boundary_D1_reduction_pct_ci_high"
            ],
    }

    for column, value in (
        replacement.items()
    ):
        old = float(
            master.at[
                idx,
                column
            ]
        )

        new = float(value)

        audit_rows.append({
            "task": "Push-T",
            "modality": "lowdim",
            "checkpoint": checkpoint,
            "field": column,
            "v3_value": old,
            "v4_value": new,
            "change": new - old,
        })

        master.at[
            idx,
            column
        ] = new


# All non-lowdim rows must remain byte-level numerically unchanged.
old = pd.read_csv(
    OLD_PATH
)

non_ld_mask = ~(
    (old["task"] == "Push-T")
    & (
        old["modality"]
        == "lowdim"
    )
)

for column in old.columns:
    a = old.loc[
        non_ld_mask,
        column,
    ].reset_index(
        drop=True
    )

    b = master.loc[
        non_ld_mask,
        column,
    ].reset_index(
        drop=True
    )

    if pd.api.types.is_numeric_dtype(a):
        if not np.array_equal(
            a.to_numpy(),
            b.to_numpy(),
            equal_nan=True,
        ):
            raise RuntimeError(
                "non-lowdim numeric row "
                f"changed: {column}"
            )
    else:
        if not a.equals(b):
            raise RuntimeError(
                "non-lowdim text row "
                f"changed: {column}"
            )


master.to_csv(
    CSV_PATH,
    index=False,
)

audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT_PATH,
    index=False,
)

with open(
    TXT_PATH,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "===== BSOP 7-CONDITION "
        "CANONICAL MASTER V4 =====\n\n"
    )
    f.write(
        "Push-T lowdim D1-D3 use "
        "formal paired common-executed-prefix "
        "semantics.\n"
    )
    f.write(
        "BlockPush and Push-T image rows are "
        "carried unchanged from v3.\n\n"
    )
    f.write(
        master.to_string(
            index=False
        )
    )
    f.write("\n")

manifest = (
    OUT
    / "artifact_sha256.txt"
)

outputs = [
    CSV_PATH,
    TXT_PATH,
    AUDIT_PATH,
]

with open(
    manifest,
    "w",
    encoding="utf-8",
) as f:
    for path in outputs:
        f.write(
            f"{sha256_file(path)}  "
            f"{path}\n"
        )

print(
    "===== BSOP 7-CONDITION "
    "CANONICAL MASTER V4 ====="
)

print(
    master.to_string(
        index=False
    )
)

print()
print(
    "===== V3 -> V4 LOWDIM AUDIT ====="
)

print(
    audit.to_string(
        index=False
    )
)

print()
print(
    "NON_LOWDIM_INVARIANCE = PASS"
)

print(
    "SUCCESS_COUNT_INVARIANCE = PASS"
)

print(
    "BSOP_7COND_MASTER_V4 = PASS"
)
