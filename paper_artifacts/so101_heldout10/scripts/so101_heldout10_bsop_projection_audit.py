from pathlib import Path
import csv
import json
import math
import hashlib

import numpy as np


# ============================================================
# 0. PATHS
# ============================================================

RAW_DIR = Path(
    "/root/autodl-tmp/so101_heldout10_boundary_export_train40_20k_v1"
)

EP_DIR = RAW_DIR / "episodes"

OUT_DIR = Path(
    "/root/autodl-tmp/so101_heldout10_bsop_projection_audit_train40_20k_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. FROZEN BSOP PARAMETERS
# ============================================================

ETA = 1.0
LAMBDA_MAX = 0.5
KAPPA = 0.5

# Frozen BEFORE observing any projected result.
#
# In checkpoint-native normalized coordinates, define the
# absolute L2 correction radius by a per-dimension RMS budget:
#
#     ||b-a||_2 <= RHO_RMS_PER_DIM * sqrt(action_dim)
#
# For d=6 and 0.10:
#     rho = 0.244948974...
#
RHO_RMS_PER_DIM = 0.10

EPS = 1e-12
TOL = 1e-8

BOOTSTRAP_SEED = 20260825
N_BOOT = 20000


# ============================================================
# 2. HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def describe(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def project_bsop(
    a,
    p,
    y,
    rho,
):
    """
    Vectorized same-boundary BSOP projection.

    a : current raw first action
    p : previous time-aligned prediction
    y : boundary reference

    All inputs are in checkpoint-native normalized space.
    """

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    q = p - a
    ay = a - y

    q_sq = np.sum(
        q * q,
        axis=-1,
    )

    q_norm = np.sqrt(
        q_sq
    )

    raw_jump = np.linalg.norm(
        ay,
        axis=-1,
    )

    # Segment minimizer.
    numer = -np.sum(
        ay * q,
        axis=-1,
    )

    lambda_star = np.clip(
        numer / (q_sq + EPS),
        0.0,
        1.0,
    )

    cap_segment = (
        ETA * lambda_star
    )

    cap_lambda = np.full_like(
        lambda_star,
        LAMBDA_MAX,
    )

    cap_rho = (
        rho
        / (q_norm + EPS)
    )

    cap_kappa = (
        KAPPA
        * raw_jump
        / (q_norm + EPS)
    )

    cap_stack = np.stack(
        [
            cap_segment,
            cap_lambda,
            cap_rho,
            cap_kappa,
        ],
        axis=-1,
    )

    lam = np.min(
        cap_stack,
        axis=-1,
    )

    lam = np.clip(
        lam,
        0.0,
        1.0,
    )

    b = (
        a
        + lam[:, None] * q
    )

    projected_jump = np.linalg.norm(
        b - y,
        axis=-1,
    )

    correction = np.linalg.norm(
        b - a,
        axis=-1,
    )

    # Relative correction w.r.t. raw boundary jump.
    relative_correction = (
        correction
        / (raw_jump + EPS)
    )

    # Fraction of p-a disagreement actually traversed.
    disagreement_fraction = (
        correction
        / (q_norm + EPS)
    )

    reduction_abs = (
        raw_jump
        - projected_jump
    )

    reduction_pct = (
        100.0
        * reduction_abs
        / (raw_jump + EPS)
    )

    active = (
        correction > TOL
    )

    # Identify the limiting cap for diagnostics.
    cap_names = np.asarray(
        [
            "segment",
            "lambda_max",
            "rho",
            "kappa",
        ],
        dtype=object,
    )

    cap_index = np.argmin(
        cap_stack,
        axis=-1,
    )

    limiting_cap = cap_names[
        cap_index
    ]

    limiting_cap = np.where(
        active,
        limiting_cap,
        "inactive",
    )

    # --------------------------------------------------------
    # Deterministic constraint checks
    # --------------------------------------------------------

    violation_nonincrease = (
        projected_jump
        > raw_jump + TOL
    )

    violation_absolute = (
        correction
        > rho + TOL
    )

    violation_relative = (
        projected_jump
        < (
            (1.0 - KAPPA)
            * raw_jump
            - TOL
        )
    )

    return {
        "b": b,
        "lambda_star": lambda_star,
        "lambda": lam,
        "q_norm": q_norm,
        "raw_jump": raw_jump,
        "projected_jump": projected_jump,
        "correction": correction,
        "relative_correction": relative_correction,
        "disagreement_fraction": disagreement_fraction,
        "reduction_abs": reduction_abs,
        "reduction_pct": reduction_pct,
        "active": active,
        "limiting_cap": limiting_cap,
        "violation_nonincrease":
            violation_nonincrease,
        "violation_absolute":
            violation_absolute,
        "violation_relative":
            violation_relative,
    }


def bootstrap_episode_cluster(
    ep_raw_sum,
    ep_proj_sum,
    ep_n,
):
    """
    Cluster bootstrap at the episode level.

    Each bootstrap draw resamples whole episodes, then
    recomputes the pooled boundary reduction.
    """

    ep_raw_sum = np.asarray(
        ep_raw_sum,
        dtype=np.float64,
    )

    ep_proj_sum = np.asarray(
        ep_proj_sum,
        dtype=np.float64,
    )

    ep_n = np.asarray(
        ep_n,
        dtype=np.int64,
    )

    n_ep = len(
        ep_raw_sum
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    vals = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for i in range(
        N_BOOT
    ):
        idx = rng.integers(
            0,
            n_ep,
            size=n_ep,
        )

        raw_total = np.sum(
            ep_raw_sum[idx]
        )

        proj_total = np.sum(
            ep_proj_sum[idx]
        )

        vals[i] = (
            100.0
            * (
                raw_total
                - proj_total
            )
            / (raw_total + EPS)
        )

    return {
        "seed":
            int(BOOTSTRAP_SEED),

        "n_boot":
            int(N_BOOT),

        "estimate_mean":
            float(np.mean(vals)),

        "ci95_low":
            float(
                np.quantile(
                    vals,
                    0.025,
                )
            ),

        "ci95_high":
            float(
                np.quantile(
                    vals,
                    0.975,
                )
            ),
    }


# ============================================================
# 3. LOAD RAW EXPORT MANIFEST
# ============================================================

manifest_path = (
    RAW_DIR
    / "manifest.json"
)

summary_path = (
    RAW_DIR
    / "summary.json"
)

assert manifest_path.exists()
assert summary_path.exists()

with open(
    manifest_path,
    "r",
) as f:
    raw_manifest = json.load(
        f
    )

with open(
    summary_path,
    "r",
) as f:
    raw_summary = json.load(
        f
    )

assert int(
    raw_summary["episodes"]
) == 10

assert int(
    raw_summary["boundaries"]
) == 263


# ============================================================
# 4. DISCOVER EPISODE FILES
# ============================================================

ep_files = sorted(
    EP_DIR.glob(
        "episode_*.npz"
    )
)

assert len(
    ep_files
) == 10

with np.load(
    ep_files[0]
) as first:

    action_dim = int(
        first[
            "a_norm"
        ].shape[-1]
    )

assert action_dim == 6

RHO = (
    RHO_RMS_PER_DIM
    * math.sqrt(
        action_dim
    )
)

print(
    "===== SO-101 BSOP NORMALIZED-SPACE AUDIT ====="
)

print(
    "raw export =",
    RAW_DIR,
)

print(
    "episodes   =",
    len(ep_files),
)

print(
    "action dim =",
    action_dim,
)

print()
print(
    "Frozen parameters:"
)

print(
    "ETA              =",
    ETA,
)

print(
    "LAMBDA_MAX       =",
    LAMBDA_MAX,
)

print(
    "KAPPA            =",
    KAPPA,
)

print(
    "RHO_RMS_PER_DIM  =",
    RHO_RMS_PER_DIM,
)

print(
    "RHO_NORM_L2      =",
    RHO,
)


# ============================================================
# 5. RUN BOTH REFERENCES
# ============================================================

references = {
    "data": "y_data_norm",
    "policy": "y_policy_norm",
}

all_boundary_rows = []

all_results = {}

total_boundaries_seen = 0


for ref_name, y_key in references.items():

    pooled = {
        "raw_jump": [],
        "projected_jump": [],
        "correction": [],
        "relative_correction": [],
        "disagreement_fraction": [],
        "reduction_abs": [],
        "reduction_pct": [],
        "lambda": [],
        "active": [],
        "q_norm": [],
        "violation_nonincrease": [],
        "violation_absolute": [],
        "violation_relative": [],
        "limiting_cap": [],
    }

    ep_raw_sum = []
    ep_proj_sum = []
    ep_n = []

    episode_reduction_pct = []
    episode_activation_pct = []

    episodes_improved = 0
    episodes_unchanged = 0
    episodes_worsened = 0

    ref_boundary_count = 0

    for ep_path in ep_files:

        with np.load(
            ep_path
        ) as z:

            ep = int(
                z[
                    "episode_index"
                ]
            )

            a = np.asarray(
                z[
                    "a_norm"
                ],
                dtype=np.float64,
            )

            p = np.asarray(
                z[
                    "p_norm"
                ],
                dtype=np.float64,
            )

            y = np.asarray(
                z[
                    y_key
                ],
                dtype=np.float64,
            )

            saved_q = np.asarray(
                z[
                    "query_disagreement_norm"
                ],
                dtype=np.float64,
            )

            prev_idx = np.asarray(
                z[
                    "boundary_prev_query_indices"
                ],
                dtype=np.int64,
            )

            curr_idx = np.asarray(
                z[
                    "boundary_curr_query_indices"
                ],
                dtype=np.int64,
            )

            assert (
                a.shape
                == p.shape
                == y.shape
            )

            assert (
                a.ndim == 2
                and a.shape[1]
                == action_dim
            )

            n = int(
                a.shape[0]
            )

            assert n > 0

            assert np.all(
                curr_idx
                - prev_idx
                == 8
            )

            result = project_bsop(
                a=a,
                p=p,
                y=y,
                rho=RHO,
            )

            # Raw exporter identity must still hold.
            assert np.allclose(
                result[
                    "q_norm"
                ],
                saved_q,
                atol=1e-6,
                rtol=1e-6,
            )

            # Check correct saved raw jump for each reference.
            if ref_name == "data":
                saved_raw = np.asarray(
                    z[
                        "raw_jump_data_ref_norm"
                    ],
                    dtype=np.float64,
                )
            else:
                saved_raw = np.asarray(
                    z[
                        "raw_jump_policy_ref_norm"
                    ],
                    dtype=np.float64,
                )

            assert np.allclose(
                result[
                    "raw_jump"
                ],
                saved_raw,
                atol=1e-6,
                rtol=1e-6,
            )

            # Formal constraints must all pass.
            assert not np.any(
                result[
                    "violation_nonincrease"
                ]
            )

            assert not np.any(
                result[
                    "violation_absolute"
                ]
            )

            assert not np.any(
                result[
                    "violation_relative"
                ]
            )

            raw_mean = float(
                np.mean(
                    result[
                        "raw_jump"
                    ]
                )
            )

            proj_mean = float(
                np.mean(
                    result[
                        "projected_jump"
                    ]
                )
            )

            ep_red = (
                100.0
                * (
                    raw_mean
                    - proj_mean
                )
                / (raw_mean + EPS)
            )

            ep_act = (
                100.0
                * np.mean(
                    result[
                        "active"
                    ]
                )
            )

            episode_reduction_pct.append(
                ep_red
            )

            episode_activation_pct.append(
                ep_act
            )

            ep_raw_sum.append(
                float(
                    np.sum(
                        result[
                            "raw_jump"
                        ]
                    )
                )
            )

            ep_proj_sum.append(
                float(
                    np.sum(
                        result[
                            "projected_jump"
                        ]
                    )
                )
            )

            ep_n.append(
                n
            )

            if proj_mean < raw_mean - TOL:
                episodes_improved += 1
            elif proj_mean > raw_mean + TOL:
                episodes_worsened += 1
            else:
                episodes_unchanged += 1

            for key in pooled:
                pooled[key].extend(
                    np.asarray(
                        result[key]
                    ).tolist()
                )

            for i in range(
                n
            ):

                all_boundary_rows.append({
                    "reference":
                        ref_name,

                    "episode":
                        ep,

                    "boundary_index":
                        i,

                    "prev_query_global_idx":
                        int(
                            prev_idx[i]
                        ),

                    "curr_query_global_idx":
                        int(
                            curr_idx[i]
                        ),

                    "query_disagreement_norm":
                        float(
                            result[
                                "q_norm"
                            ][i]
                        ),

                    "raw_jump_norm":
                        float(
                            result[
                                "raw_jump"
                            ][i]
                        ),

                    "projected_jump_norm":
                        float(
                            result[
                                "projected_jump"
                            ][i]
                        ),

                    "reduction_abs_norm":
                        float(
                            result[
                                "reduction_abs"
                            ][i]
                        ),

                    "reduction_pct":
                        float(
                            result[
                                "reduction_pct"
                            ][i]
                        ),

                    "correction_norm":
                        float(
                            result[
                                "correction"
                            ][i]
                        ),

                    "relative_correction":
                        float(
                            result[
                                "relative_correction"
                            ][i]
                        ),

                    "disagreement_fraction":
                        float(
                            result[
                                "disagreement_fraction"
                            ][i]
                        ),

                    "lambda":
                        float(
                            result[
                                "lambda"
                            ][i]
                        ),

                    "active":
                        int(
                            result[
                                "active"
                            ][i]
                        ),

                    "limiting_cap":
                        str(
                            result[
                                "limiting_cap"
                            ][i]
                        ),

                    "violation_nonincrease":
                        int(
                            result[
                                "violation_nonincrease"
                            ][i]
                        ),

                    "violation_absolute":
                        int(
                            result[
                                "violation_absolute"
                            ][i]
                        ),

                    "violation_relative":
                        int(
                            result[
                                "violation_relative"
                            ][i]
                        ),
                })

            ref_boundary_count += n

    assert ref_boundary_count == 263

    raw_arr = np.asarray(
        pooled[
            "raw_jump"
        ],
        dtype=np.float64,
    )

    proj_arr = np.asarray(
        pooled[
            "projected_jump"
        ],
        dtype=np.float64,
    )

    corr_arr = np.asarray(
        pooled[
            "correction"
        ],
        dtype=np.float64,
    )

    rel_corr_arr = np.asarray(
        pooled[
            "relative_correction"
        ],
        dtype=np.float64,
    )

    active_arr = np.asarray(
        pooled[
            "active"
        ],
        dtype=bool,
    )

    red_pct_arr = np.asarray(
        pooled[
            "reduction_pct"
        ],
        dtype=np.float64,
    )

    pooled_reduction_pct = (
        100.0
        * (
            np.mean(raw_arr)
            - np.mean(proj_arr)
        )
        / (
            np.mean(raw_arr)
            + EPS
        )
    )

    bootstrap = (
        bootstrap_episode_cluster(
            ep_raw_sum,
            ep_proj_sum,
            ep_n,
        )
    )

    limiting = np.asarray(
        pooled[
            "limiting_cap"
        ],
        dtype=object,
    )

    cap_counts = {}

    for cap in [
        "inactive",
        "segment",
        "lambda_max",
        "rho",
        "kappa",
    ]:
        cap_counts[cap] = int(
            np.sum(
                limiting == cap
            )
        )

    ref_summary = {
        "reference":
            ref_name,

        "boundaries":
            int(
                raw_arr.size
            ),

        "raw_jump_norm":
            describe(
                raw_arr
            ),

        "projected_jump_norm":
            describe(
                proj_arr
            ),

        "correction_norm":
            describe(
                corr_arr
            ),

        "relative_correction":
            describe(
                rel_corr_arr
            ),

        "boundary_reduction_pct":
            describe(
                red_pct_arr
            ),

        "pooled_reduction_pct":
            float(
                pooled_reduction_pct
            ),

        "activation_rate_pct":
            float(
                100.0
                * np.mean(
                    active_arr
                )
            ),

        "active_boundaries":
            int(
                np.sum(
                    active_arr
                )
            ),

        "episodes_improved":
            int(
                episodes_improved
            ),

        "episodes_unchanged":
            int(
                episodes_unchanged
            ),

        "episodes_worsened":
            int(
                episodes_worsened
            ),

        "episode_reduction_pct":
            describe(
                episode_reduction_pct
            ),

        "episode_activation_pct":
            describe(
                episode_activation_pct
            ),

        "cluster_bootstrap_reduction_pct":
            bootstrap,

        "limiting_cap_counts":
            cap_counts,

        "violations": {
            "nonincrease":
                int(
                    np.sum(
                        pooled[
                            "violation_nonincrease"
                        ]
                    )
                ),

            "absolute":
                int(
                    np.sum(
                        pooled[
                            "violation_absolute"
                        ]
                    )
                ),

            "relative":
                int(
                    np.sum(
                        pooled[
                            "violation_relative"
                        ]
                    )
                ),
        },
    }

    all_results[
        ref_name
    ] = ref_summary

    total_boundaries_seen = (
        ref_boundary_count
    )


# ============================================================
# 6. SAVE BOUNDARY CSV
# ============================================================

csv_path = (
    OUT_DIR
    / "projection_boundaries.csv"
)

fieldnames = [
    "reference",
    "episode",
    "boundary_index",
    "prev_query_global_idx",
    "curr_query_global_idx",
    "query_disagreement_norm",
    "raw_jump_norm",
    "projected_jump_norm",
    "reduction_abs_norm",
    "reduction_pct",
    "correction_norm",
    "relative_correction",
    "disagreement_fraction",
    "lambda",
    "active",
    "limiting_cap",
    "violation_nonincrease",
    "violation_absolute",
    "violation_relative",
]

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_boundary_rows
    )


# ============================================================
# 7. SAVE SUMMARY + MANIFEST
# ============================================================

audit_summary = {
    "source_raw_export":
        str(
            RAW_DIR
        ),

    "episodes":
        10,

    "boundaries_per_reference":
        int(
            total_boundaries_seen
        ),

    "action_dim":
        int(
            action_dim
        ),

    "space":
        (
            "checkpoint-native normalized action coordinates"
        ),

    "parameters": {
        "eta":
            ETA,

        "lambda_max":
            LAMBDA_MAX,

        "kappa":
            KAPPA,

        "rho_rms_per_dim":
            RHO_RMS_PER_DIM,

        "rho_norm_l2":
            RHO,

        "rho_rule":
            (
                "rho_norm_l2 = "
                "rho_rms_per_dim * sqrt(action_dim)"
            ),
    },

    "primary_reference":
        (
            "data: recorded robot action immediately "
            "preceding the current query"
        ),

    "supplementary_reference":
        (
            "policy: previous query's last executable "
            "prediction; not actually executed under "
            "teacher forcing"
        ),

    "data_reference":
        all_results[
            "data"
        ],

    "policy_reference":
        all_results[
            "policy"
        ],
}

audit_summary_path = (
    OUT_DIR
    / "summary.json"
)

with open(
    audit_summary_path,
    "w",
) as f:

    json.dump(
        audit_summary,
        f,
        indent=2,
    )


audit_manifest = {
    "source_manifest":
        str(
            manifest_path
        ),

    "source_manifest_sha256":
        sha256_file(
            manifest_path
        ),

    "source_summary":
        str(
            summary_path
        ),

    "source_summary_sha256":
        sha256_file(
            summary_path
        ),

    "projection_summary":
        str(
            audit_summary_path
        ),

    "projection_csv":
        str(
            csv_path
        ),

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "n_bootstrap":
        N_BOOT,

    "note":
        (
            "No model inference was rerun. "
            "Projection was applied offline to the "
            "frozen raw SO-101 boundary export."
        ),
}

audit_manifest_path = (
    OUT_DIR
    / "manifest.json"
)

with open(
    audit_manifest_path,
    "w",
) as f:

    json.dump(
        audit_manifest,
        f,
        indent=2,
    )


# ============================================================
# 8. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("SO-101 BSOP NORMALIZED-SPACE PROJECTION AUDIT COMPLETE")
print("=" * 78)

print(
    f"episodes                 = {len(ep_files)}"
)

print(
    f"boundaries per reference = {total_boundaries_seen}"
)

print(
    f"rho_norm_l2              = {RHO:.9f}"
)

print(
    f"rho_rms_per_dim          = {RHO_RMS_PER_DIM:.6f}"
)


for ref_name in [
    "data",
    "policy",
]:

    s = all_results[
        ref_name
    ]

    print()
    print(
        f"----- {ref_name.upper()} REFERENCE -----"
    )

    print(
        "raw jump mean            = "
        f"{s['raw_jump_norm']['mean']:.6f}"
    )

    print(
        "projected jump mean      = "
        f"{s['projected_jump_norm']['mean']:.6f}"
    )

    print(
        "pooled reduction         = "
        f"{s['pooled_reduction_pct']:.3f}%"
    )

    boot = s[
        "cluster_bootstrap_reduction_pct"
    ]

    print(
        "episode-cluster 95% CI   = "
        f"[{boot['ci95_low']:.3f}%, "
        f"{boot['ci95_high']:.3f}%]"
    )

    print(
        "activation rate          = "
        f"{s['activation_rate_pct']:.3f}%"
    )

    print(
        "correction mean          = "
        f"{s['correction_norm']['mean']:.6f}"
    )

    print(
        "relative correction mean = "
        f"{s['relative_correction']['mean']:.6f}"
    )

    print(
        "relative correction p95  = "
        f"{s['relative_correction']['p95']:.6f}"
    )

    print(
        "episodes improved        = "
        f"{s['episodes_improved']}/{len(ep_files)}"
    )

    print(
        "episodes unchanged       = "
        f"{s['episodes_unchanged']}/{len(ep_files)}"
    )

    print(
        "episodes worsened        = "
        f"{s['episodes_worsened']}/{len(ep_files)}"
    )

    print(
        "limiting caps            = "
        f"{s['limiting_cap_counts']}"
    )

    print(
        "violations nonincrease   = "
        f"{s['violations']['nonincrease']}"
    )

    print(
        "violations absolute      = "
        f"{s['violations']['absolute']}"
    )

    print(
        "violations relative      = "
        f"{s['violations']['relative']}"
    )


assert (
    all_results[
        "data"
    ][
        "violations"
    ][
        "nonincrease"
    ]
    == 0
)

assert (
    all_results[
        "data"
    ][
        "violations"
    ][
        "absolute"
    ]
    == 0
)

assert (
    all_results[
        "data"
    ][
        "violations"
    ][
        "relative"
    ]
    == 0
)

assert (
    all_results[
        "policy"
    ][
        "violations"
    ][
        "nonincrease"
    ]
    == 0
)

assert (
    all_results[
        "policy"
    ][
        "violations"
    ][
        "absolute"
    ]
    == 0
)

assert (
    all_results[
        "policy"
    ][
        "violations"
    ][
        "relative"
    ]
    == 0
)


print()
print("Saved:")
print(" CSV      =", csv_path)
print(" summary  =", audit_summary_path)
print(" manifest =", audit_manifest_path)

print()
print(
    "PRIMARY PAPER REFERENCE: y_data"
)

print(
    "SUPPLEMENTARY DIAGNOSTIC: y_policy"
)

print(
    "ALL DETERMINISTIC CONSTRAINT CHECKS: PASS"
)

print("=" * 78)
