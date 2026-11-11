from pathlib import Path
import csv, json, hashlib, math
import numpy as np

RUN = "blockpush_bsop_aasf_confirmatory_fresh50"
TRACE = Path("paper_artifacts/traces") / RUN
RESULT = Path("paper_exp/results") / RUN
OUT = Path("paper_exp/tables") / RUN
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["original", "aasf", "bsop", "bsop_aasf"]
SEEDS = list(range(4400100, 4400150))
BOOT_N = 20000
BOOT_SEED = 20260808


def load_score(method):
    with (RESULT / method / "eval_log.json").open() as f:
        d = json.load(f)

    return {
        s: float(d[f"test/sim_max_reward_{s}"])
        for s in SEEDS
    }


def load_trace(method, seed):
    paths = list(
        (TRACE / method).glob(
            f"seed_{seed}_*.npz"
        )
    )

    if len(paths) != 1:
        raise RuntimeError(
            f"{method} seed={seed}: "
            f"expected 1 trace, got {len(paths)}"
        )

    with np.load(
        paths[0],
        allow_pickle=False,
    ) as d:

        out = {
            "action": np.asarray(
                d["actual_executed_action"],
                dtype=np.float64,
            ),
            "steps": int(
                d["actual_executed_steps"]
            ),
            "executed": np.asarray(
                d["executed_steps_per_query"],
                dtype=np.int64,
            ),
            "version": str(
                d["execution_trace_version"]
            ),
        }

        if method in {
            "bsop",
            "bsop_aasf",
        }:
            for key in [
                "bsop_boundary_executed",
                "bsop_raw_boundary_jump",
                "bsop_projected_boundary_jump",
                "bsop_correction",
                "bsop_relative_correction",
                "bsop_active",
            ]:
                out[key] = np.asarray(
                    d[key]
                ).copy()

            out["bsop_rho"] = float(
                d["bsop_rho"]
            )
            out["bsop_kappa"] = float(
                d["bsop_kappa"]
            )

    return paths[0], out


def dv(action, order):
    if len(action) <= order:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    return np.linalg.norm(
        np.diff(
            action,
            n=order,
            axis=0,
        ),
        axis=-1,
    )


def mean(values):
    return (
        float(np.mean(values))
        if len(values)
        else float("nan")
    )


def boundary(action, counts):
    values = []
    cursor = 0
    previous_last = None

    for count in counts:
        count = int(count)

        if count == 0:
            continue

        current = action[
            cursor:cursor + count
        ]

        if len(current) != count:
            raise RuntimeError(
                "boundary accounting overflow"
            )

        if previous_last is not None:
            values.append(
                float(
                    np.linalg.norm(
                        current[0]
                        - previous_last
                    )
                )
            )

        previous_last = (
            current[-1].copy()
        )
        cursor += count

    if cursor != len(action):
        raise RuntimeError(
            "boundary accounting mismatch: "
            f"{cursor} != {len(action)}"
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def reduction(base, new):
    if (
        not np.isfinite(base)
        or not np.isfinite(new)
        or abs(base) <= 1e-12
    ):
        return float("nan")

    return (
        100.0
        * (base - new)
        / base
    )


def boot_ci(values, seed):
    x = np.asarray(
        values,
        dtype=np.float64,
    )
    x = x[np.isfinite(x)]

    if not len(x):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    means = np.empty(
        BOOT_N,
        dtype=np.float64,
    )

    for start in range(
        0,
        BOOT_N,
        2000,
    ):
        n = min(
            2000,
            BOOT_N - start,
        )

        idx = rng.integers(
            0,
            len(x),
            size=(n, len(x)),
        )

        means[
            start:start + n
        ] = x[idx].mean(axis=1)

    return (
        float(x.mean()),
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def sign_test(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )
    x = x[np.isfinite(x)]

    pos = int(
        np.sum(x > 0)
    )
    neg = int(
        np.sum(x < 0)
    )

    n = pos + neg

    if n == 0:
        return pos, neg, 1.0

    k = min(pos, neg)

    p = (
        2.0
        * sum(
            math.comb(n, i)
            for i in range(k + 1)
        )
        / (2 ** n)
    )

    return (
        pos,
        neg,
        min(1.0, p),
    )


scores = {
    method: load_score(method)
    for method in METHODS
}

data = {
    method: {}
    for method in METHODS
}

per_seed = []

for method in METHODS:
    for seed in SEEDS:

        path, trace = load_trace(
            method,
            seed,
        )

        if (
            trace["version"]
            != "actual_execution_trace_v1"
        ):
            raise RuntimeError(
                f"{method} {seed}: "
                "bad trace version"
            )

        if (
            len(trace["action"])
            != trace["steps"]
        ):
            raise RuntimeError(
                f"{method} {seed}: "
                "action/step mismatch"
            )

        d1 = dv(
            trace["action"],
            1,
        )
        d2 = dv(
            trace["action"],
            2,
        )
        d3 = dv(
            trace["action"],
            3,
        )

        bd = boundary(
            trace["action"],
            trace["executed"],
        )

        score = scores[
            method
        ][seed]

        row = {
            "method": method,
            "seed": seed,
            "score": score,
            "p1": int(
                score > 0.4
            ),
            "p2": int(
                score > 0.9
            ),
            "steps": trace["steps"],
            "d1_mean": mean(d1),
            "d2_mean": mean(d2),
            "d3_mean": mean(d3),
            "d1_sum": float(
                d1.sum()
            ),
            "d2_sum": float(
                d2.sum()
            ),
            "d3_sum": float(
                d3.sum()
            ),
            "boundary_d1_mean": mean(
                bd
            ),
            "boundary_count": len(
                bd
            ),
            "trace": str(path),
        }

        per_seed.append(row)

        trace["metrics"] = row

        data[
            method
        ][seed] = trace


PER_SEED = (
    OUT
    / "blockpush_per_seed_metrics.csv"
)

with PER_SEED.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            per_seed[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        per_seed
    )


def summary(method):
    rows = [
        data[
            method
        ][seed]["metrics"]
        for seed in SEEDS
    ]

    return {
        "score": np.mean(
            [
                x["score"]
                for x in rows
            ]
        ),
        "p1": sum(
            x["p1"]
            for x in rows
        ),
        "p2": sum(
            x["p2"]
            for x in rows
        ),
        "steps": np.mean(
            [
                x["steps"]
                for x in rows
            ]
        ),
        "d1": np.mean(
            [
                x["d1_mean"]
                for x in rows
            ]
        ),
        "d2": np.mean(
            [
                x["d2_mean"]
                for x in rows
            ]
        ),
        "d3": np.mean(
            [
                x["d3_mean"]
                for x in rows
            ]
        ),
        "boundary": np.nanmean(
            [
                x[
                    "boundary_d1_mean"
                ]
                for x in rows
            ]
        ),
    }


S = {
    method: summary(method)
    for method in METHODS
}


pairs = [
    ("original", "aasf"),
    ("original", "bsop"),
    ("aasf", "bsop_aasf"),
    ("original", "bsop_aasf"),
]

contrast_rows = []
contrast_text = []

for pair_index, (
    base,
    new,
) in enumerate(pairs):

    values = {
        "score_difference": [],
        "executed_step_difference": [],
        "common_prefix_d1_reduction_pct": [],
        "common_prefix_d2_reduction_pct": [],
        "common_prefix_d3_reduction_pct": [],
        "full_cumulative_d1_reduction_pct": [],
        "full_cumulative_d2_reduction_pct": [],
        "full_cumulative_d3_reduction_pct": [],
        "own_trajectory_boundary_d1_reduction_pct": [],
    }

    for seed in SEEDS:

        b = data[base][seed]
        n = data[new][seed]

        ba = b["action"]
        na = n["action"]

        common = min(
            len(ba),
            len(na),
        )

        values[
            "score_difference"
        ].append(
            n["metrics"]["score"]
            - b["metrics"]["score"]
        )

        values[
            "executed_step_difference"
        ].append(
            n["metrics"]["steps"]
            - b["metrics"]["steps"]
        )

        for order in [
            1,
            2,
            3,
        ]:

            values[
                f"common_prefix_d{order}_reduction_pct"
            ].append(
                reduction(
                    mean(
                        dv(
                            ba[:common],
                            order,
                        )
                    ),
                    mean(
                        dv(
                            na[:common],
                            order,
                        )
                    ),
                )
            )

            values[
                f"full_cumulative_d{order}_reduction_pct"
            ].append(
                reduction(
                    float(
                        dv(
                            ba,
                            order,
                        ).sum()
                    ),
                    float(
                        dv(
                            na,
                            order,
                        ).sum()
                    ),
                )
            )

        values[
            "own_trajectory_boundary_d1_reduction_pct"
        ].append(
            reduction(
                b["metrics"][
                    "boundary_d1_mean"
                ],
                n["metrics"][
                    "boundary_d1_mean"
                ],
            )
        )

    contrast_text.append(
        f"\n--- {base} -> {new} ---\n"
        f"P1: "
        f"{S[base]['p1']}/50 -> "
        f"{S[new]['p1']}/50\n"
        f"P2: "
        f"{S[base]['p2']}/50 -> "
        f"{S[new]['p2']}/50"
    )

    for metric_index, (
        metric,
        vals,
    ) in enumerate(
        values.items()
    ):

        mu, lo, hi = boot_ci(
            vals,
            BOOT_SEED
            + pair_index * 100
            + metric_index,
        )

        finite = np.asarray(
            vals,
            dtype=np.float64,
        )
        finite = finite[
            np.isfinite(finite)
        ]

        median = (
            float(
                np.median(finite)
            )
            if len(finite)
            else float("nan")
        )

        pos, neg, p = sign_test(
            vals
        )

        contrast_rows.append({
            "base": base,
            "new": new,
            "metric": metric,
            "mean": mu,
            "ci95_low": lo,
            "ci95_high": hi,
            "median": median,
            "positive": pos,
            "negative": neg,
            "exact_sign_p": p,
        })

        suffix = (
            "%"
            if "reduction_pct"
            in metric
            else ""
        )

        contrast_text.append(
            f"{metric}: "
            f"mean={mu:+.4f}{suffix}, "
            f"95% CI="
            f"[{lo:+.4f}, "
            f"{hi:+.4f}]{suffix}, "
            f"median="
            f"{median:+.4f}{suffix}, "
            f"positive/negative="
            f"{pos}/{neg}, "
            f"sign_p={p:.6g}"
        )


CONTRAST = (
    OUT
    / "blockpush_paired_contrasts.csv"
)

with CONTRAST.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            contrast_rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        contrast_rows
    )


mechanism_rows = []
mechanism_text = []

for method_index, method in enumerate(
    [
        "bsop",
        "bsop_aasf",
    ]
):

    RAW = []
    PROJ = []
    CORR = []
    REL = []
    ACTIVE = []

    seed_reductions = []

    boundary_viol = 0
    absolute_viol = 0
    relative_viol = 0

    for seed in SEEDS:

        trace = data[
            method
        ][seed]

        mask = np.asarray(
            trace[
                "bsop_boundary_executed"
            ],
            dtype=bool,
        )

        raw = np.asarray(
            trace[
                "bsop_raw_boundary_jump"
            ],
            dtype=float,
        )

        projected = np.asarray(
            trace[
                "bsop_projected_boundary_jump"
            ],
            dtype=float,
        )

        correction = np.asarray(
            trace[
                "bsop_correction"
            ],
            dtype=float,
        )

        relative = np.asarray(
            trace[
                "bsop_relative_correction"
            ],
            dtype=float,
        )

        active = np.asarray(
            trace[
                "bsop_active"
            ],
            dtype=bool,
        )

        valid = (
            mask
            & np.isfinite(raw)
            & np.isfinite(projected)
            & np.isfinite(correction)
            & np.isfinite(relative)
        )

        raw = raw[valid]
        projected = projected[valid]
        correction = correction[valid]
        relative = relative[valid]
        active = active[valid]

        seed_reduction = reduction(
            raw.mean(),
            projected.mean(),
        )

        seed_reductions.append(
            seed_reduction
        )

        bv = int(
            np.sum(
                projected
                > raw + 1e-4
            )
        )

        av = int(
            np.sum(
                correction
                > trace[
                    "bsop_rho"
                ] + 1e-4
            )
        )

        rv = int(
            np.sum(
                correction
                > (
                    trace[
                        "bsop_kappa"
                    ]
                    * raw
                    + 1e-4
                )
            )
        )

        boundary_viol += bv
        absolute_viol += av
        relative_viol += rv

        RAW.append(raw)
        PROJ.append(projected)
        CORR.append(correction)
        REL.append(relative)
        ACTIVE.append(active)

        mechanism_rows.append({
            "method": method,
            "seed": seed,
            "executed_boundaries": len(
                raw
            ),
            "active_boundaries": int(
                active.sum()
            ),
            "same_boundary_reduction_pct": (
                seed_reduction
            ),
            "raw_boundary_mean": float(
                raw.mean()
            ),
            "projected_boundary_mean": float(
                projected.mean()
            ),
            "correction_mean": float(
                correction.mean()
            ),
            "relative_correction_mean": float(
                relative.mean()
            ),
            "boundary_violations": bv,
            "absolute_cap_violations": av,
            "relative_cap_violations": rv,
        })

    raw = np.concatenate(RAW)
    projected = np.concatenate(PROJ)
    correction = np.concatenate(CORR)
    relative = np.concatenate(REL)
    active = np.concatenate(ACTIVE)

    mu, lo, hi = boot_ci(
        seed_reductions,
        BOOT_SEED
        + 9000
        + method_index,
    )

    mechanism_text.append(
        f"\n===== "
        f"{method.upper()} "
        f"SAME-BOUNDARY MECHANISM =====\n"
        f"Transitions: {len(raw)}\n"
        f"Active transitions: "
        f"{100 * active.mean():.2f}%\n"
        f"Correction mean/P95/max: "
        f"{correction.mean():.9f} / "
        f"{np.percentile(correction, 95):.9f} / "
        f"{correction.max():.9f}\n"
        f"Relative correction "
        f"median/P95/max: "
        f"{np.median(relative):.6f} / "
        f"{np.percentile(relative, 95):.6f} / "
        f"{relative.max():.6f}\n"
        f"Raw/projected boundary mean: "
        f"{raw.mean():.9f} / "
        f"{projected.mean():.9f}\n"
        f"Pooled same-boundary reduction: "
        f"{reduction(raw.mean(), projected.mean()):.4f}%\n"
        f"Per-seed same-boundary reduction: "
        f"mean={mu:.4f}%, "
        f"95% CI=[{lo:.4f}, {hi:.4f}]%, "
        f"improved="
        f"{int(np.sum(np.asarray(seed_reductions) > 0))}/50\n"
        f"Violations "
        f"(boundary/absolute/relative): "
        f"{boundary_viol}/"
        f"{absolute_viol}/"
        f"{relative_viol}"
    )


MECHANISM = (
    OUT
    / "blockpush_bsop_mechanism.csv"
)

with MECHANISM.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            mechanism_rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        mechanism_rows
    )


lines = [
    "===== BLOCKPUSH CONFIRMATORY FRESH-50 SUMMARY =====",
    "",
    "===== FOUR-METHOD ABSOLUTE SUMMARY =====",
    (
        f"{'method':<12}"
        f"{'score':>9}"
        f"{'p1':>9}"
        f"{'p2':>9}"
        f"{'steps':>9}"
        f"{'D1':>11}"
        f"{'D2':>11}"
        f"{'D3':>11}"
        f"{'boundaryD1':>13}"
    ),
]

for method in METHODS:
    x = S[method]

    lines.append(
        f"{method:<12}"
        f"{x['score']:>9.4f}"
        f"{x['p1']:>6d}/50"
        f"{x['p2']:>6d}/50"
        f"{x['steps']:>9.1f}"
        f"{x['d1']:>11.6f}"
        f"{x['d2']:>11.6f}"
        f"{x['d3']:>11.6f}"
        f"{x['boundary']:>13.6f}"
    )

lines += [
    "",
    "===== PAIRED CONTRASTS =====",
]

lines += contrast_text
lines += mechanism_text


SUMMARY = (
    OUT
    / "blockpush_summary.txt"
)

SUMMARY.write_text(
    "\n".join(lines) + "\n"
)

print(
    SUMMARY.read_text()
)

print(
    "\n===== ANALYSIS OUTPUTS ====="
)

for path in [
    PER_SEED,
    CONTRAST,
    MECHANISM,
    SUMMARY,
]:
    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    print(
        digest,
        path,
    )
