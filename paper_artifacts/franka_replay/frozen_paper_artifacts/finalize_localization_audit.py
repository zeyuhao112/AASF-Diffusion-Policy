from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/root/autodl-tmp/bsop_franka_replay/"
    "formal_replay_v1/per_trajectory"
)

OUT = Path(
    "/root/autodl-tmp/bsop_franka_replay/"
    "formal_replay_v1/boundary_dynamics"
)
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "original",
    "aasf",
    "bsop",
    "bsop_aasf",
]

K = 8
B = 20000
RNG_SEED = 20260824

rows = []

def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))

for seed in range(4300100, 4300150):
    for method in METHODS:

        path = ROOT / f"{method}_seed{seed}_timeseries.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        z = np.load(path, allow_pickle=True)

        cmd = np.asarray(z["command_index"], dtype=int)
        tr = np.asarray(z["torque_rate_norm"], dtype=float)
        jerk = np.asarray(z["ee_jerk_norm"], dtype=float)

        if not (len(cmd) == len(tr) == len(jerk)):
            raise RuntimeError(
                f"length mismatch: {seed} {method}"
            )

        # Formal replay starts from command_index=1.
        # New chunk first commands are 8, 16, 24, ...
        boundary = (cmd % K) == 0
        nonboundary = ~boundary

        if not np.any(boundary):
            raise RuntimeError(
                f"No boundary samples: {seed} {method}"
            )

        if not np.any(nonboundary):
            raise RuntimeError(
                f"No nonboundary samples: {seed} {method}"
            )

        rows.append({
            "seed": seed,
            "method": method,

            "boundary_samples":
                int(boundary.sum()),

            "nonboundary_samples":
                int(nonboundary.sum()),

            "boundary_commands":
                int(len(np.unique(cmd[boundary]))),

            "boundary_torque_rate_rms":
                rms(tr[boundary]),

            "boundary_torque_rate_p95":
                float(np.percentile(
                    tr[boundary], 95
                )),

            "boundary_jerk_rms":
                rms(jerk[boundary]),

            "boundary_jerk_p95":
                float(np.percentile(
                    jerk[boundary], 95
                )),

            "nonboundary_torque_rate_rms":
                rms(tr[nonboundary]),

            "nonboundary_torque_rate_p95":
                float(np.percentile(
                    tr[nonboundary], 95
                )),

            "nonboundary_jerk_rms":
                rms(jerk[nonboundary]),

            "nonboundary_jerk_p95":
                float(np.percentile(
                    jerk[nonboundary], 95
                )),
        })

df = pd.DataFrame(rows)

PER_SEED_OUT = (
    OUT / "boundary_nonboundary_localization_per_seed.csv"
)
df.to_csv(PER_SEED_OUT, index=False)

assert len(df) == 200
assert df["seed"].nunique() == 50
assert set(df["method"]) == set(METHODS)

wide = {
    col: df.pivot(
        index="seed",
        columns="method",
        values=col
    )
    for col in [
        "boundary_torque_rate_rms",
        "boundary_torque_rate_p95",
        "boundary_jerk_rms",
        "boundary_jerk_p95",
        "nonboundary_torque_rate_rms",
        "nonboundary_torque_rate_p95",
        "nonboundary_jerk_rms",
        "nonboundary_jerk_p95",
    ]
}

comparisons = [
    ("original", "bsop"),
    ("aasf", "bsop_aasf"),
]

metric_pairs = [
    (
        "torque_rate_rms",
        "boundary_torque_rate_rms",
        "nonboundary_torque_rate_rms",
    ),
    (
        "torque_rate_p95",
        "boundary_torque_rate_p95",
        "nonboundary_torque_rate_p95",
    ),
    (
        "jerk_rms",
        "boundary_jerk_rms",
        "nonboundary_jerk_rms",
    ),
    (
        "jerk_p95",
        "boundary_jerk_p95",
        "nonboundary_jerk_p95",
    ),
]

rng = np.random.default_rng(RNG_SEED)
summary_rows = []

for base, method in comparisons:

    for label, bmetric, nmetric in metric_pairs:

        bx = wide[bmetric][base].to_numpy(dtype=float)
        by = wide[bmetric][method].to_numpy(dtype=float)

        nx = wide[nmetric][base].to_numpy(dtype=float)
        ny = wide[nmetric][method].to_numpy(dtype=float)

        if not (
            len(bx) == len(by) ==
            len(nx) == len(ny) == 50
        ):
            raise RuntimeError(
                f"pairing failure: {base}->{method} {label}"
            )

        if np.any(bx <= 0) or np.any(nx <= 0):
            raise RuntimeError(
                f"non-positive baseline: {base} {label}"
            )

        # Positive = lower dynamics under method
        b_red = 100.0 * (bx - by) / bx
        n_red = 100.0 * (nx - ny) / nx

        # Positive = reduction is stronger at boundary
        gap = b_red - n_red

        # Same paired seed resampling for all quantities
        idx = rng.integers(
            0,
            50,
            size=(B, 50)
        )

        boot_b = b_red[idx].mean(axis=1)
        boot_n = n_red[idx].mean(axis=1)
        boot_g = gap[idx].mean(axis=1)

        b_lo, b_hi = np.percentile(
            boot_b, [2.5, 97.5]
        )
        n_lo, n_hi = np.percentile(
            boot_n, [2.5, 97.5]
        )
        g_lo, g_hi = np.percentile(
            boot_g, [2.5, 97.5]
        )

        summary_rows.append({
            "base": base,
            "method": method,
            "metric": label,

            "boundary_reduction_pct":
                b_red.mean(),

            "boundary_ci_low_pct":
                b_lo,

            "boundary_ci_high_pct":
                b_hi,

            "nonboundary_reduction_pct":
                n_red.mean(),

            "nonboundary_ci_low_pct":
                n_lo,

            "nonboundary_ci_high_pct":
                n_hi,

            "localization_gap_pct_points":
                gap.mean(),

            "localization_gap_ci_low":
                g_lo,

            "localization_gap_ci_high":
                g_hi,

            "boundary_improved_seeds":
                int(np.sum(by < bx)),

            "nonboundary_improved_seeds":
                int(np.sum(ny < nx)),

            "gap_positive_seeds":
                int(np.sum(gap > 0)),
        })

summary = pd.DataFrame(summary_rows)

SUMMARY_OUT = (
    OUT / "boundary_nonboundary_localization_summary.csv"
)
summary.to_csv(SUMMARY_OUT, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

for base, method in comparisons:

    print(
        "\n============================================================"
    )
    print(
        f"LOCALIZATION: {base.upper()} -> {method.upper()}"
    )
    print(
        "============================================================"
    )

    tmp = summary[
        (summary["base"] == base)
        & (summary["method"] == method)
    ]

    print(
        tmp[
            [
                "metric",
                "boundary_reduction_pct",
                "boundary_ci_low_pct",
                "boundary_ci_high_pct",
                "nonboundary_reduction_pct",
                "nonboundary_ci_low_pct",
                "nonboundary_ci_high_pct",
                "localization_gap_pct_points",
                "localization_gap_ci_low",
                "localization_gap_ci_high",
                "gap_positive_seeds",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(PER_SEED_OUT)
print(SUMMARY_OUT)
print("\nPASS: boundary/nonboundary localization audit completed.")
