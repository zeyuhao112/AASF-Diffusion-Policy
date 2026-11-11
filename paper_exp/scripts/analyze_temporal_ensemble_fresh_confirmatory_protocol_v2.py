from pathlib import Path
import csv
import hashlib
import subprocess
import numpy as np

ROOT = Path("paper_exp/temporal_ensemble_fresh_seed42")
TRACE = ROOT / "formal_confirmatory" / "traces"
OUT = ROOT / "formal_confirmatory_protocol_analysis_v2"

CONF_MAN = (
    ROOT / "formal_confirmatory"
    / "TE_FRESH_CONFIRMATORY_ARTIFACT_SHA256.txt"
)
OLD_MAN = (
    ROOT / "formal_confirmatory_analysis"
    / "analysis_artifact_sha256.txt"
)

A_PATH = Path(
    "paper_exp/scripts/"
    "analyze_temporal_ensemble_fresh_confirmatory_protocol_v2.py"
)

REFERENCE = Path(
    "paper_exp/scripts/"
    "analyze_savgol_confirmatory_seed42.py"
)

SEEDS = range(4800100, 4800150)
DT = 0.1

BOOTSTRAP_SAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260816


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_manifest(path):
    r = subprocess.run(
        ["sha256sum", "-c", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Manifest verification failed: {path}"
        )


def fd(x, order, dt=1.0):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise RuntimeError(
            f"trajectory shape invalid: {x.shape}"
        )

    if order < 1 or dt <= 0:
        raise RuntimeError("invalid order/dt")

    for _ in range(order):
        if x.shape[0] < 2:
            raise RuntimeError(
                "trajectory too short"
            )
        x = np.diff(x, axis=0)

    return float(
        np.mean(
            np.linalg.norm(x, axis=-1)
        )
        / (float(dt) ** order)
    )


def reduction(original, te, name):
    original = float(original)
    te = float(te)

    if (
        not np.isfinite(original)
        or not np.isfinite(te)
        or original <= 0
    ):
        raise RuntimeError(
            f"{name}: invalid values"
        )

    return float(
        100.0 * (original - te) / original
    )


def bootstrap_ci(values, seed):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise RuntimeError(
            "bootstrap input must be 1D"
        )

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "non-finite bootstrap input"
        )

    n = values.size

    rng = np.random.default_rng(
        int(seed)
    )

    means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    done = 0
    batch = 5000

    while done < BOOTSTRAP_SAMPLES:
        m = min(
            batch,
            BOOTSTRAP_SAMPLES - done,
        )

        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        means[done:done+m] = np.mean(
            values[idx],
            axis=1,
        )

        done += m

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


def load_action(method, seed):
    p = (
        TRACE
        / method
        / f"test_seed_{seed}.npz"
    )

    if not p.is_file():
        raise RuntimeError(
            f"missing trace: {p}"
        )

    with np.load(
        p,
        allow_pickle=False,
    ) as z:
        for key in (
            "seed",
            "executed_action",
            "actual_executed_steps",
        ):
            if key not in z.files:
                raise RuntimeError(
                    f"{p}: missing {key}"
                )

        trace_seed = int(
            np.asarray(
                z["seed"]
            ).reshape(())
        )

        if trace_seed != seed:
            raise RuntimeError(
                f"{p}: seed mismatch"
            )

        action = np.asarray(
            z["executed_action"],
            dtype=np.float64,
        )

        n = int(
            np.asarray(
                z["actual_executed_steps"]
            ).reshape(())
        )

        if (
            action.ndim != 2
            or action.shape[0] != n
        ):
            raise RuntimeError(
                f"{p}: execution mismatch"
            )

        return action


verify_manifest(CONF_MAN)
verify_manifest(OLD_MAN)

if OUT.exists():
    raise RuntimeError(
        "Refusing to overwrite v2 analysis"
    )

rows = []

for seed in SEEDS:
    o = load_action(
        "original",
        seed,
    )

    t = load_action(
        "te_beta_2p0",
        seed,
    )

    n = min(
        o.shape[0],
        t.shape[0],
    )

    if n < 4:
        raise RuntimeError(
            f"seed {seed}: common prefix < 4"
        )

    o = o[:n]
    t = t[:n]

    row = {
        "seed": seed,
        "common_prefix_steps": n,
    }

    for order in (1, 2, 3):
        od = fd(
            o,
            order,
            dt=1.0,
        )
        td = fd(
            t,
            order,
            dt=1.0,
        )

        oc = fd(
            o,
            order,
            dt=DT,
        )
        tc = fd(
            t,
            order,
            dt=DT,
        )

        dr = reduction(
            od,
            td,
            f"D{order}",
        )

        cr = reduction(
            oc,
            tc,
            f"C{order}",
        )

        if not np.isclose(
            dr,
            cr,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"seed {seed}: "
                f"D{order}/C{order} "
                "reduction invariance failed"
            )

        row[f"original_D{order}"] = od
        row[f"te_D{order}"] = td
        row[f"D{order}_reduction_pct"] = dr

        row[f"original_C{order}"] = oc
        row[f"te_C{order}"] = tc
        row[f"C{order}_reduction_pct"] = cr

    rows.append(row)


OUT.mkdir(
    parents=False,
    exist_ok=False,
)

per_seed = OUT / "protocol_v2_per_seed.csv"

with open(
    per_seed,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)


stats = []

for order in (1, 2, 3):
    vals = np.asarray(
        [
            r[
                f"D{order}_reduction_pct"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    mean = float(
        np.mean(vals)
    )

    lo, hi = bootstrap_ci(
        vals,
        BOOTSTRAP_BASE_SEED + order,
    )

    stats.append(
        {
            "metric": f"D{order}",
            "mean_reduction_pct": mean,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "bootstrap_seed":
                BOOTSTRAP_BASE_SEED + order,
        }
    )

    # Ck percentage reduction is mathematically
    # identical to Dk for fixed dt.
    stats.append(
        {
            "metric": f"C{order}",
            "mean_reduction_pct": mean,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "bootstrap_seed":
                BOOTSTRAP_BASE_SEED + order,
        }
    )


stats_path = (
    OUT / "protocol_v2_paired_stats.csv"
)

with open(
    stats_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(stats[0]),
    )
    writer.writeheader()
    writer.writerows(stats)


by_metric = {
    r["metric"]: r
    for r in stats
}

prefix = np.asarray(
    [
        r["common_prefix_steps"]
        for r in rows
    ],
    dtype=np.int64,
)

table = OUT / "protocol_v2_table.txt"

with open(
    table,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "TEMPORAL ENSEMBLE FORMAL "
        "PROTOCOL-CONFORMANT SMOOTHNESS\n"
    )
    f.write(
        "==============================================\n\n"
    )

    f.write(
        "Comparison: original vs te_beta_2p0\n"
    )
    f.write(
        "Seeds: 4800100-4800149 (n=50)\n"
    )
    f.write(
        "Common-prefix rule: paired executed prefix\n"
    )
    f.write(
        "control_dt_s = 0.1\n\n"
    )

    for order in (1, 2, 3):
        r = by_metric[f"D{order}"]

        f.write(
            f"D{order} reduction = "
            f"{r['mean_reduction_pct']:+.6f}%\n"
        )
        f.write(
            "  95% CI "
            f"[{r['bootstrap_ci95_low']:+.6f}, "
            f"{r['bootstrap_ci95_high']:+.6f}]\n"
        )

        f.write(
            f"C{order} reduction = "
            f"{r['mean_reduction_pct']:+.6f}%\n"
        )
        f.write(
            "  95% CI "
            f"[{r['bootstrap_ci95_low']:+.6f}, "
            f"{r['bootstrap_ci95_high']:+.6f}]\n\n"
        )

    f.write(
        "common_prefix_steps: "
        f"min={int(prefix.min())}, "
        f"mean={float(prefix.mean()):.3f}, "
        f"max={int(prefix.max())}\n"
    )


audit = OUT / "protocol_v2_audit.txt"

with open(
    audit,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "TEMPORAL ENSEMBLE FORMAL "
        "PROTOCOL V2 AUDIT\n"
    )
    f.write(
        "====================================\n\n"
    )

    f.write(
        "analysis_scope="
        "supplemental_offline_only\n"
    )
    f.write(
        "formal_rollout_repeated=NO\n"
    )
    f.write(
        "methods=original,te_beta_2p0\n"
    )
    f.write(
        "frozen_beta=2.0\n"
    )
    f.write(
        "seeds=4800100-4800149\n"
    )
    f.write(
        "paired_common_prefix=YES\n"
    )
    f.write(
        "Boundary_D1_recomputed=NO\n"
    )
    f.write(
        "control_dt_s=0.1\n"
    )
    f.write(
        "D1_D2_D3_dt=1\n"
    )
    f.write(
        "C1_C2_C3_time_normalized=YES\n"
    )
    f.write(
        "D_C_reduction_invariance=PASS\n"
    )
    f.write(
        f"bootstrap_samples={BOOTSTRAP_SAMPLES}\n"
    )
    f.write(
        f"bootstrap_base_seed={BOOTSTRAP_BASE_SEED}\n"
    )
    f.write(
        "alternative_beta_evaluation=NO\n"
    )
    f.write(
        "post_unblind_retuning=NO\n"
    )
    f.write(
        "confirmatory_manifest_verification=PASS\n"
    )
    f.write(
        "prior_analysis_manifest_verification=PASS\n"
    )

    if REFERENCE.is_file():
        f.write(
            "canonical_reference_sha256="
            f"{sha(REFERENCE)}\n"
        )

    f.write(
        f"analyzer_sha256={sha(A_PATH)}\n"
    )

    f.write(
        "\n"
        "TE_PROTOCOL_V2_ANALYSIS_AUDIT = PASS\n"
    )


manifest = (
    OUT / "analysis_artifact_sha256.txt"
)

with open(
    manifest,
    "w",
    encoding="utf-8",
) as f:
    for p in (
        A_PATH,
        CONF_MAN,
        OLD_MAN,
        per_seed,
        stats_path,
        table,
        audit,
    ):
        f.write(
            f"{sha(p)}  {p}\n"
        )


verify_manifest(manifest)

print(
    "TE_PROTOCOL_V2_ANALYSIS_AUDIT = PASS"
)
print(
    "TE_PROTOCOL_V2_ANALYSIS_FINAL = PASS"
)
print(
    f"analysis_output = {OUT}"
)
print()
print(table.read_text(encoding="utf-8"))
