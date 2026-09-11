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
PHYSICS_DT = 0.002
STEPS_PER_CMD = 50

rows = []

for seed in range(4300100, 4300150):

    for method in METHODS:

        path = (
            ROOT
            / f"{method}_seed{seed}_timeseries.npz"
        )

        z = np.load(path, allow_pickle=True)

        cmd = np.asarray(z["command_index"], dtype=int)
        tr = np.asarray(z["torque_rate_norm"], dtype=float)
        jerk = np.asarray(z["ee_jerk_norm"], dtype=float)

        # First action of every new 8-action chunk.
        # cmd=0 was initial condition and is not in formal samples.
        boundary = (
            (cmd % K) == 0
        )

        nonboundary = ~boundary

        if not np.any(boundary):
            raise RuntimeError(
                f"No boundary samples: {seed} {method}"
            )

        rows.append({
            "seed": seed,
            "method": method,

            "boundary_samples":
                int(boundary.sum()),

            "boundary_commands":
                int(len(np.unique(cmd[boundary]))),

            "boundary_torque_rate_rms":
                float(np.sqrt(
                    np.mean(tr[boundary] ** 2)
                )),

            "boundary_torque_rate_p95":
                float(np.percentile(
                    tr[boundary], 95
                )),

            "boundary_jerk_rms":
                float(np.sqrt(
                    np.mean(jerk[boundary] ** 2)
                )),

            "boundary_jerk_p95":
                float(np.percentile(
                    jerk[boundary], 95
                )),

            "nonboundary_torque_rate_rms":
                float(np.sqrt(
                    np.mean(tr[nonboundary] ** 2)
                )),

            "nonboundary_jerk_rms":
                float(np.sqrt(
                    np.mean(jerk[nonboundary] ** 2)
                )),
        })


df = pd.DataFrame(rows)

out = OUT / "boundary_dynamics_per_seed.csv"
df.to_csv(out, index=False)

print("===== BOUNDARY DYNAMICS MEANS =====")

cols = [
    "boundary_torque_rate_rms",
    "boundary_torque_rate_p95",
    "boundary_jerk_rms",
    "boundary_jerk_p95",
]

print(
    df.groupby("method")[cols]
      .mean()
      .to_string()
)

print("\n===== PAIRED MEAN REDUCTIONS =====")

wide = df.pivot(
    index="seed",
    columns="method"
)

comparisons = [
    ("original", "bsop"),
    ("aasf", "bsop_aasf"),
    ("original", "bsop_aasf"),
]

for base, method in comparisons:

    print(f"\n{base} -> {method}")

    for metric in cols:

        x = wide[metric][base].values
        y = wide[metric][method].values

        red = 100.0 * (x-y) / x

        print(
            metric,
            f"{red.mean():.3f}% ",
            f"improved {np.sum(y < x)}/50"
        )

print("\nSaved:", out)
print("PASS: boundary dynamics audit completed.")
