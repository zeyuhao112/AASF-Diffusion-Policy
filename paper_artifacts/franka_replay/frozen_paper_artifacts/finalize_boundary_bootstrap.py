from pathlib import Path
import numpy as np
import pandas as pd

IN = Path(
    "/root/autodl-tmp/bsop_franka_replay/"
    "formal_replay_v1/boundary_dynamics/"
    "boundary_dynamics_per_seed.csv"
)

OUT = IN.parent / "boundary_dynamics_paired_bootstrap.csv"

df = pd.read_csv(IN)

required_methods = {"original", "aasf", "bsop", "bsop_aasf"}
assert set(df["method"]) == required_methods, set(df["method"])
assert df["seed"].nunique() == 50
assert len(df) == 200

metrics = [
    "boundary_torque_rate_rms",
    "boundary_torque_rate_p95",
    "boundary_jerk_rms",
    "boundary_jerk_p95",
]

comparisons = [
    ("original", "bsop", "PRIMARY_BSOP"),
    ("aasf", "bsop_aasf", "PRIMARY_BSOP_ON_AASF"),
    ("original", "bsop_aasf", "SECONDARY_COMBINED"),
]

wide = {
    metric: df.pivot(index="seed", columns="method", values=metric)
    for metric in metrics
}

rng = np.random.default_rng(20260824)
B = 20000
rows = []

for base, method, role in comparisons:
    for metric in metrics:
        w = wide[metric]

        # Paired seeds only
        pair = w[[base, method]].dropna()

        x = pair[base].to_numpy(dtype=float)
        y = pair[method].to_numpy(dtype=float)

        if len(x) != 50:
            raise RuntimeError(
                f"{base}->{method} {metric}: "
                f"expected 50 paired seeds, got {len(x)}"
            )

        if np.any(x <= 0):
            raise RuntimeError(
                f"non-positive baseline encountered: {base} {metric}"
            )

        # Positive = method is better / lower
        reduction = 100.0 * (x - y) / x

        # Paired bootstrap over seeds
        idx = rng.integers(
            0, len(reduction),
            size=(B, len(reduction))
        )
        boot = reduction[idx].mean(axis=1)

        ci_low, ci_high = np.percentile(
            boot, [2.5, 97.5]
        )

        # Ratio of absolute method means, useful diagnostic
        ratio_of_means = (
            100.0 * (x.mean() - y.mean()) / x.mean()
        )

        rows.append({
            "role": role,
            "base": base,
            "method": method,
            "metric": metric,
            "n_pairs": len(x),

            "base_mean": x.mean(),
            "method_mean": y.mean(),

            "mean_per_seed_reduction_pct":
                reduction.mean(),

            "median_per_seed_reduction_pct":
                np.median(reduction),

            "bootstrap_ci_low_pct":
                ci_low,

            "bootstrap_ci_high_pct":
                ci_high,

            "ratio_of_means_reduction_pct":
                ratio_of_means,

            "improved_seeds":
                int(np.sum(y < x)),

            "unchanged_seeds":
                int(np.sum(np.isclose(y, x))),

            "worsened_seeds":
                int(np.sum(y > x)),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print("\n===== PRIMARY: ORIGINAL -> BSOP =====")
print(
    out[out["role"] == "PRIMARY_BSOP"][
        [
            "metric",
            "base_mean",
            "method_mean",
            "mean_per_seed_reduction_pct",
            "median_per_seed_reduction_pct",
            "bootstrap_ci_low_pct",
            "bootstrap_ci_high_pct",
            "ratio_of_means_reduction_pct",
            "improved_seeds",
            "worsened_seeds",
        ]
    ].to_string(index=False)
)

print("\n===== PRIMARY: AASF -> BSOP+AASF =====")
print(
    out[out["role"] == "PRIMARY_BSOP_ON_AASF"][
        [
            "metric",
            "base_mean",
            "method_mean",
            "mean_per_seed_reduction_pct",
            "median_per_seed_reduction_pct",
            "bootstrap_ci_low_pct",
            "bootstrap_ci_high_pct",
            "ratio_of_means_reduction_pct",
            "improved_seeds",
            "worsened_seeds",
        ]
    ].to_string(index=False)
)

print("\n===== SECONDARY: ORIGINAL -> BSOP+AASF =====")
print(
    out[out["role"] == "SECONDARY_COMBINED"][
        [
            "metric",
            "mean_per_seed_reduction_pct",
            "median_per_seed_reduction_pct",
            "bootstrap_ci_low_pct",
            "bootstrap_ci_high_pct",
            "ratio_of_means_reduction_pct",
            "improved_seeds",
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUT)
print("PASS: paired bootstrap completed.")
