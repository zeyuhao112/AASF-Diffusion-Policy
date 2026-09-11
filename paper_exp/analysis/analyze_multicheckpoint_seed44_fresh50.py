from pathlib import Path
import csv
import hashlib
import math
import numpy as np


RUN_NAME = "bsop_aasf_multicheckpoint_seed44_fresh50"

TRACE_ROOT = Path(
    "paper_artifacts/traces"
) / RUN_NAME

OUTPUT_ROOT = Path(
    "paper_exp/tables"
) / RUN_NAME

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

EXPECTED_SEEDS = set(
    range(4300100, 4300150)
)

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260805

PER_SEED_PATH = (
    OUTPUT_ROOT
    / "confirmatory_per_seed_metrics.csv"
)

CONTRAST_PATH = (
    OUTPUT_ROOT
    / "confirmatory_paired_contrasts.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "confirmatory_summary.txt"
)


def load_npz(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def load_method(root):
    normal = {}
    diagnostic = {}

    for path in sorted(
        root.glob("*.npz")
    ):
        data = load_npz(path)
        seed = int(data["seed"])

        if path.name.endswith(
            "_overlap_diag.npz"
        ):
            if seed in diagnostic:
                raise RuntimeError(
                    f"Duplicate diagnostic seed "
                    f"{seed} in {root}"
                )
            diagnostic[seed] = data
        else:
            if seed in normal:
                raise RuntimeError(
                    f"Duplicate normal seed "
                    f"{seed} in {root}"
                )
            normal[seed] = data

    if set(normal) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{root}: normal seed mismatch: "
            f"{sorted(normal)}"
        )

    if set(diagnostic) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{root}: diagnostic seed mismatch: "
            f"{sorted(diagnostic)}"
        )

    for seed, trace in normal.items():
        version = str(
            trace["execution_trace_version"]
        )

        if version != (
            "actual_execution_trace_v1"
        ):
            raise RuntimeError(
                f"Seed {seed}: unexpected "
                f"execution version {version}"
            )

        action = np.asarray(
            trace["executed_action"]
        )
        reward = np.asarray(
            trace["reward_history"]
        )
        executed_steps = int(
            trace["actual_executed_steps"]
        )

        if action.shape[0] != executed_steps:
            raise RuntimeError(
                f"Seed {seed}: action length "
                f"does not match executed count"
            )

        if reward.shape[0] != executed_steps:
            raise RuntimeError(
                f"Seed {seed}: reward length "
                f"does not match executed count"
            )

    return normal, diagnostic


def diff_values(action, order):
    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if action.shape[0] <= order:
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


def diff_metrics(action, order):
    values = diff_values(
        action,
        order,
    )

    if values.size == 0:
        return {
            "mean": float("nan"),
            "sum": 0.0,
            "count": 0,
        }

    return {
        "mean": float(np.mean(values)),
        "sum": float(np.sum(values)),
        "count": int(values.size),
    }


def first_reward_step(reward, threshold):
    reward = np.asarray(
        reward,
        dtype=np.float64,
    )

    indices = np.flatnonzero(
        reward >= threshold
    )

    if indices.size == 0:
        return -1

    return int(indices[0] + 1)


def query_starts(trace):
    counts = np.asarray(
        trace["executed_steps_per_query"],
        dtype=np.int64,
    )

    if counts.ndim != 1:
        raise RuntimeError(
            "executed_steps_per_query "
            "must be one-dimensional"
        )

    if np.any(counts <= 0):
        raise RuntimeError(
            "Every recorded query must execute "
            "at least one action"
        )

    return np.concatenate([
        np.asarray(
            [0],
            dtype=np.int64,
        ),
        np.cumsum(
            counts[:-1],
            dtype=np.int64,
        ),
    ])


def boundary_metrics(trace):
    action = np.asarray(
        trace["executed_action"],
        dtype=np.float64,
    )

    starts = query_starts(trace)

    d1 = diff_values(
        action,
        1,
    )

    indices = starts[1:] - 1
    indices = indices[
        (indices >= 0)
        & (indices < d1.size)
    ]

    mask = np.zeros(
        d1.size,
        dtype=bool,
    )
    mask[indices] = True

    boundary = d1[mask]
    internal = d1[~mask]

    return {
        "count": int(boundary.size),
        "mean": (
            float(np.mean(boundary))
            if boundary.size
            else float("nan")
        ),
        "sum": float(np.sum(boundary)),
        "internal_mean": (
            float(np.mean(internal))
            if internal.size
            else float("nan")
        ),
    }


def safe_reduction(old_value, new_value):
    if not (
        np.isfinite(old_value)
        and np.isfinite(new_value)
    ):
        return float("nan")

    if abs(old_value) < 1.0e-12:
        return float("nan")

    return float(
        100.0
        * (
            1.0
            - new_value / old_value
        )
    )


def bootstrap_mean_ci(
    values,
    samples,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )

    boot = np.mean(
        values[indices],
        axis=1,
    )

    return (
        float(np.mean(values)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def exact_sign_pvalue(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    values = values[
        np.isfinite(values)
    ]

    positive = int(
        np.sum(values > 0)
    )
    negative = int(
        np.sum(values < 0)
    )
    n = positive + negative

    if n == 0:
        return 1.0

    k = min(
        positive,
        negative,
    )

    probability = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2.0 ** n)

    return float(
        min(1.0, 2.0 * probability)
    )


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


traces = {}
diagnostics = {}

for method in METHODS:
    normal, diagnostic = load_method(
        TRACE_ROOT / method
    )
    traces[method] = normal
    diagnostics[method] = diagnostic


# ---------------------------------------------------------
# Per-method, per-seed metrics
# ---------------------------------------------------------
per_seed_rows = []

for method in METHODS:
    for seed in sorted(
        EXPECTED_SEEDS
    ):
        trace = traces[method][seed]

        action = np.asarray(
            trace["executed_action"],
            dtype=np.float64,
        )
        reward = np.asarray(
            trace["reward_history"],
            dtype=np.float64,
        )

        d1 = diff_metrics(action, 1)
        d2 = diff_metrics(action, 2)
        d3 = diff_metrics(action, 3)
        boundary = boundary_metrics(trace)

        score = float(trace["score"])
        success = bool(
            trace[
                "terminated_before_time_limit"
            ]
        )

        row = {
            "method": method,
            "seed": seed,
            "score": score,
            "sr_090": int(
                score >= 0.90
            ),
            "exact_success": int(success),
            "actual_executed_steps": int(
                trace[
                    "actual_executed_steps"
                ]
            ),
            "queries": int(
                np.asarray(
                    trace[
                        "executed_steps_per_query"
                    ]
                ).size
            ),
            "first_reward_090": (
                first_reward_step(
                    reward,
                    0.90,
                )
            ),
            "first_reward_095": (
                first_reward_step(
                    reward,
                    0.95,
                )
            ),
            "first_reward_100": (
                first_reward_step(
                    reward,
                    1.00,
                )
            ),
            "D1_mean": d1["mean"],
            "D1_total": d1["sum"],
            "D2_mean": d2["mean"],
            "D2_total": d2["sum"],
            "D3_mean": d3["mean"],
            "D3_total": d3["sum"],
            "boundary_D1_mean": (
                boundary["mean"]
            ),
            "boundary_D1_total": (
                boundary["sum"]
            ),
            "boundary_count": (
                boundary["count"]
            ),
            "internal_D1_mean": (
                boundary["internal_mean"]
            ),
        }

        per_seed_rows.append(row)

write_csv(
    PER_SEED_PATH,
    per_seed_rows,
)


# ---------------------------------------------------------
# Paired contrasts
# ---------------------------------------------------------
CONTRASTS = [
    (
        "original",
        "aasf",
        "component",
    ),
    (
        "original",
        "bsop",
        "component",
    ),
    (
        "aasf",
        "bsop_aasf",
        "primary",
    ),
    (
        "original",
        "bsop_aasf",
        "primary",
    ),
    (
        "bsop",
        "bsop_aasf",
        "component",
    ),
]

contrast_rows = []
contrast_arrays = {}


for old_method, new_method, role in CONTRASTS:
    contrast_name = (
        f"{old_method}_to_{new_method}"
    )

    contrast_arrays[
        contrast_name
    ] = {
        "score_difference": [],
        "step_difference": [],
        "common_D1_reduction_pct": [],
        "common_D2_reduction_pct": [],
        "common_D3_reduction_pct": [],
        "full_D1_total_reduction_pct": [],
        "full_D2_total_reduction_pct": [],
        "full_D3_total_reduction_pct": [],
        "boundary_D1_reduction_pct": [],
        "success_difference": [],
        "sr090_difference": [],
    }

    for seed in sorted(
        EXPECTED_SEEDS
    ):
        old = traces[
            old_method
        ][seed]
        new = traces[
            new_method
        ][seed]

        old_action = np.asarray(
            old["executed_action"],
            dtype=np.float64,
        )
        new_action = np.asarray(
            new["executed_action"],
            dtype=np.float64,
        )

        common_length = min(
            old_action.shape[0],
            new_action.shape[0],
        )

        old_common = old_action[
            :common_length
        ]
        new_common = new_action[
            :common_length
        ]

        common_metrics = {}

        for order, metric in [
            (1, "D1"),
            (2, "D2"),
            (3, "D3"),
        ]:
            old_value = diff_metrics(
                old_common,
                order,
            )["mean"]
            new_value = diff_metrics(
                new_common,
                order,
            )["mean"]

            common_metrics[
                metric
            ] = safe_reduction(
                old_value,
                new_value,
            )

        full_total = {}

        for order, metric in [
            (1, "D1"),
            (2, "D2"),
            (3, "D3"),
        ]:
            old_value = diff_metrics(
                old_action,
                order,
            )["sum"]
            new_value = diff_metrics(
                new_action,
                order,
            )["sum"]

            full_total[
                metric
            ] = safe_reduction(
                old_value,
                new_value,
            )

        old_boundary = (
            boundary_metrics(old)["mean"]
        )
        new_boundary = (
            boundary_metrics(new)["mean"]
        )

        boundary_reduction = (
            safe_reduction(
                old_boundary,
                new_boundary,
            )
        )

        old_score = float(
            old["score"]
        )
        new_score = float(
            new["score"]
        )

        old_success = int(bool(
            old[
                "terminated_before_time_limit"
            ]
        ))
        new_success = int(bool(
            new[
                "terminated_before_time_limit"
            ]
        ))

        old_steps = int(
            old[
                "actual_executed_steps"
            ]
        )
        new_steps = int(
            new[
                "actual_executed_steps"
            ]
        )

        row = {
            "contrast": contrast_name,
            "role": role,
            "old_method": old_method,
            "new_method": new_method,
            "seed": seed,
            "old_score": old_score,
            "new_score": new_score,
            "score_difference": (
                new_score - old_score
            ),
            "old_success": old_success,
            "new_success": new_success,
            "success_difference": (
                new_success - old_success
            ),
            "old_sr090": int(
                old_score >= 0.90
            ),
            "new_sr090": int(
                new_score >= 0.90
            ),
            "sr090_difference": (
                int(new_score >= 0.90)
                - int(old_score >= 0.90)
            ),
            "old_steps": old_steps,
            "new_steps": new_steps,
            "step_difference": (
                new_steps - old_steps
            ),
            "common_executed_length": (
                common_length
            ),
            "common_D1_reduction_pct": (
                common_metrics["D1"]
            ),
            "common_D2_reduction_pct": (
                common_metrics["D2"]
            ),
            "common_D3_reduction_pct": (
                common_metrics["D3"]
            ),
            "full_D1_total_reduction_pct": (
                full_total["D1"]
            ),
            "full_D2_total_reduction_pct": (
                full_total["D2"]
            ),
            "full_D3_total_reduction_pct": (
                full_total["D3"]
            ),
            "boundary_D1_reduction_pct": (
                boundary_reduction
            ),
        }

        contrast_rows.append(row)

        arrays = contrast_arrays[
            contrast_name
        ]

        for key in arrays:
            arrays[key].append(
                row[key]
            )

write_csv(
    CONTRAST_PATH,
    contrast_rows,
)


# ---------------------------------------------------------
# Same-boundary BSOP+AASF mechanism verification
# ---------------------------------------------------------
mechanism_rows = []
mechanism_seed_reductions = []

boundary_violations = 0
absolute_cap_violations = 0
relative_cap_violations = 0

all_raw_jump = []
all_projected_jump = []
all_correction = []
all_relative = []


for seed in sorted(
    EXPECTED_SEEDS
):
    trace = traces[
        "bsop_aasf"
    ][seed]
    diagnostic = diagnostics[
        "bsop_aasf"
    ][seed]

    action = np.asarray(
        trace["executed_action"],
        dtype=np.float64,
    )

    starts = query_starts(
        trace
    )[1:]

    current = np.asarray(
        diagnostic[
            "current_overlap_prediction"
        ],
        dtype=np.float64,
    )

    if current.shape[0] != starts.size:
        raise RuntimeError(
            f"Seed {seed}: diagnostic "
            f"transition mismatch: "
            f"{current.shape[0]} vs "
            f"{starts.size}"
        )

    y = action[starts - 1]
    b = action[starts]
    a = current[:, 0]

    raw_jump = np.linalg.norm(
        a - y,
        axis=-1,
    )
    projected_jump = np.linalg.norm(
        b - y,
        axis=-1,
    )
    correction = np.linalg.norm(
        b - a,
        axis=-1,
    )
    relative = correction / (
        raw_jump + 1.0e-12
    )

    boundary_violations += int(
        np.sum(
            projected_jump
            > raw_jump + 1.0e-4
        )
    )
    absolute_cap_violations += int(
        np.sum(
            correction
            > 5.0 + 1.0e-4
        )
    )
    relative_cap_violations += int(
        np.sum(
            correction
            > 0.5 * raw_jump
            + 1.0e-4
        )
    )

    seed_reduction = (
        100.0
        * (
            1.0
            - np.mean(projected_jump)
            / (
                np.mean(raw_jump)
                + 1.0e-12
            )
        )
    )

    mechanism_seed_reductions.append(
        seed_reduction
    )

    for index in range(
        starts.size
    ):
        mechanism_rows.append({
            "seed": seed,
            "transition_index": index,
            "executed_step": int(
                starts[index] + 1
            ),
            "raw_boundary_jump": float(
                raw_jump[index]
            ),
            "projected_boundary_jump": float(
                projected_jump[index]
            ),
            "boundary_reduction": float(
                raw_jump[index]
                - projected_jump[index]
            ),
            "correction": float(
                correction[index]
            ),
            "relative_correction": float(
                relative[index]
            ),
            "active": int(
                correction[index]
                > 1.0e-8
            ),
        })

    all_raw_jump.append(raw_jump)
    all_projected_jump.append(
        projected_jump
    )
    all_correction.append(correction)
    all_relative.append(relative)


MECHANISM_PATH = (
    OUTPUT_ROOT
    / "bsop_aasf_boundary_mechanism.csv"
)

write_csv(
    MECHANISM_PATH,
    mechanism_rows,
)

all_raw_jump = np.concatenate(
    all_raw_jump
)
all_projected_jump = np.concatenate(
    all_projected_jump
)
all_correction = np.concatenate(
    all_correction
)
all_relative = np.concatenate(
    all_relative
)


if (
    boundary_violations
    + absolute_cap_violations
    + relative_cap_violations
) != 0:
    raise RuntimeError(
        "At least one BSOP safety "
        "guarantee was violated"
    )


# ---------------------------------------------------------
# Text summary
# ---------------------------------------------------------
lines = []


def emit(text=""):
    print(text)
    lines.append(text)


emit(
    "===== FRESH-50 CONFIRMATORY SUMMARY ====="
)
emit(
    "Seeds: 4300100-4300149"
)
emit(
    "All smoothness metrics use strict "
    "actual executed actions."
)
emit()

emit(
    "===== FOUR-METHOD ABSOLUTE SUMMARY ====="
)
emit(
    "method        score    SR@0.90  success"
    "   steps      D1      D2      D3"
    "   boundaryD1"
)

for method in METHODS:
    rows = [
        row
        for row in per_seed_rows
        if row["method"] == method
    ]

    score = np.asarray([
        row["score"]
        for row in rows
    ])
    sr090 = np.asarray([
        row["sr_090"]
        for row in rows
    ])
    success = np.asarray([
        row["exact_success"]
        for row in rows
    ])
    steps = np.asarray([
        row["actual_executed_steps"]
        for row in rows
    ])
    d1 = np.asarray([
        row["D1_mean"]
        for row in rows
    ])
    d2 = np.asarray([
        row["D2_mean"]
        for row in rows
    ])
    d3 = np.asarray([
        row["D3_mean"]
        for row in rows
    ])
    boundary = np.asarray([
        row["boundary_D1_mean"]
        for row in rows
    ])

    emit(
        f"{method:12s}"
        f" {np.mean(score):7.4f}"
        f" {np.sum(sr090):8d}/50"
        f" {np.sum(success):7d}/50"
        f" {np.mean(steps):7.1f}"
        f" {np.mean(d1):7.3f}"
        f" {np.mean(d2):7.3f}"
        f" {np.mean(d3):7.3f}"
        f" {np.mean(boundary):11.3f}"
    )


emit()
emit(
    "===== PAIRED CONTRASTS ====="
)
emit(
    "Positive smoothness reduction means "
    "the new method is smoother."
)

summary_keys = [
    (
        "score_difference",
        "Score difference",
        "difference",
    ),
    (
        "step_difference",
        "Executed-step difference",
        "difference",
    ),
    (
        "common_D1_reduction_pct",
        "Common-prefix D1 reduction",
        "reduction",
    ),
    (
        "common_D2_reduction_pct",
        "Common-prefix D2 reduction",
        "reduction",
    ),
    (
        "common_D3_reduction_pct",
        "Common-prefix D3 reduction",
        "reduction",
    ),
    (
        "full_D1_total_reduction_pct",
        "Full cumulative D1 reduction",
        "reduction",
    ),
    (
        "full_D2_total_reduction_pct",
        "Full cumulative D2 reduction",
        "reduction",
    ),
    (
        "full_D3_total_reduction_pct",
        "Full cumulative D3 reduction",
        "reduction",
    ),
    (
        "boundary_D1_reduction_pct",
        "Own-trajectory boundary D1 reduction",
        "reduction",
    ),
]


for contrast_index, (
    old_method,
    new_method,
    role,
) in enumerate(CONTRASTS):
    contrast_name = (
        f"{old_method}_to_{new_method}"
    )
    arrays = contrast_arrays[
        contrast_name
    ]

    emit()
    emit(
        f"--- {old_method} -> "
        f"{new_method} ({role}) ---"
    )

    old_rows = [
        row
        for row in per_seed_rows
        if row["method"] == old_method
    ]
    new_rows = [
        row
        for row in per_seed_rows
        if row["method"] == new_method
    ]

    old_success = sum(
        row["exact_success"]
        for row in old_rows
    )
    new_success = sum(
        row["exact_success"]
        for row in new_rows
    )
    old_sr = sum(
        row["sr_090"]
        for row in old_rows
    )
    new_sr = sum(
        row["sr_090"]
        for row in new_rows
    )

    emit(
        f"SR@0.90: {old_sr}/50 -> "
        f"{new_sr}/50"
    )
    emit(
        f"Exact success: "
        f"{old_success}/50 -> "
        f"{new_success}/50"
    )

    for key_index, (
        key,
        label,
        kind,
    ) in enumerate(summary_keys):
        values = np.asarray(
            arrays[key],
            dtype=np.float64,
        )

        mean_value, ci_low, ci_high = (
            bootstrap_mean_ci(
                values,
                BOOTSTRAP_SAMPLES,
                BOOTSTRAP_SEED
                + contrast_index * 100
                + key_index,
            )
        )

        improved = int(
            np.sum(values > 0)
        )
        worsened = int(
            np.sum(values < 0)
        )
        sign_p = exact_sign_pvalue(
            values
        )

        suffix = (
            "%"
            if kind == "reduction"
            else ""
        )

        emit(
            f"{label}: "
            f"mean={mean_value:+.4f}{suffix}, "
            f"95% bootstrap CI="
            f"[{ci_low:+.4f}, "
            f"{ci_high:+.4f}]{suffix}, "
            f"median="
            f"{np.nanmedian(values):+.4f}"
            f"{suffix}, "
            f"positive/negative="
            f"{improved}/{worsened}, "
            f"exact sign p="
            f"{sign_p:.6g}"
        )


emit()
emit(
    "===== BSOP+AASF SAME-BOUNDARY "
    "MECHANISM ====="
)
emit(
    f"Transitions: "
    f"{all_correction.size}"
)
emit(
    "Active transitions: "
    f"{100.0*np.mean(all_correction > 1e-8):.2f}%"
)
emit(
    "Correction mean/P95/max: "
    f"{np.mean(all_correction):.6f} / "
    f"{np.percentile(all_correction,95):.6f} / "
    f"{np.max(all_correction):.6f}"
)
emit(
    "Relative correction "
    "median/P95/max: "
    f"{np.median(all_relative):.6f} / "
    f"{np.percentile(all_relative,95):.6f} / "
    f"{np.max(all_relative):.6f}"
)
emit(
    "Raw/projected boundary mean: "
    f"{np.mean(all_raw_jump):.6f} / "
    f"{np.mean(all_projected_jump):.6f}"
)
pooled_same_boundary_reduction = (
    100.0
    * (
        1.0
        - np.mean(all_projected_jump)
        / (
            np.mean(all_raw_jump)
            + 1.0e-12
        )
    )
)

emit(
    "Pooled same-boundary reduction: "
    f"{pooled_same_boundary_reduction:.4f}%"
)

mechanism_mean, mechanism_low, mechanism_high = (
    bootstrap_mean_ci(
        mechanism_seed_reductions,
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED + 9999,
    )
)

emit(
    "Per-seed same-boundary reduction: "
    f"mean={mechanism_mean:.4f}%, "
    f"95% bootstrap CI="
    f"[{mechanism_low:.4f}, "
    f"{mechanism_high:.4f}]%, "
    f"improved="
    f"{np.sum(np.asarray(mechanism_seed_reductions)>0)}/50"
)
emit(
    "Violations "
    "(boundary/absolute/relative): "
    f"{boundary_violations}/"
    f"{absolute_cap_violations}/"
    f"{relative_cap_violations}"
)

SUMMARY_PATH.write_text(
    "\n".join(lines) + "\n"
)


# ---------------------------------------------------------
# Output hashes
# ---------------------------------------------------------
emit()
emit(
    "===== ANALYSIS OUTPUTS ====="
)

for path in [
    PER_SEED_PATH,
    CONTRAST_PATH,
    MECHANISM_PATH,
    SUMMARY_PATH,
]:
    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    emit(
        f"{digest}  {path}"
    )
