import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================
# Frozen protocol
# ============================================================

SEEDS = list(range(4700000, 4700050))
N_BOOTSTRAP = 100000
BOOTSTRAP_SEED = 20260714
TIE_TOL = 1e-12

RUN = "pusht_image_aasf_confirmatory_fresh50"

TRACE_ROOT = Path("paper_artifacts/traces") / RUN
RESULT_ROOT = Path("paper_exp/results") / RUN

OUT_DIR = Path("paper_exp/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "pusht_image_aasf_confirmatory_fresh50"

EPISODE_CSV = OUT_DIR / f"{PREFIX}_episode_metrics.csv"
TASK_CSV = OUT_DIR / f"{PREFIX}_task_statistics.csv"
SMOOTH_CSV = OUT_DIR / f"{PREFIX}_smoothness_statistics.csv"
MECH_CSV = OUT_DIR / f"{PREFIX}_mechanism_statistics.csv"
JSON_OUT = OUT_DIR / f"{PREFIX}_statistics.json"
REPORT_OUT = OUT_DIR / f"{PREFIX}_report.txt"

METHODS = ("original", "aasf")
METRICS = {"D1": 1, "D2": 2, "D3": 3}
SCOPES = ("common_prefix", "full_trace")

SUCCESS_THRESHOLDS = (0.90, 0.95)

ALPHA_MIN = 0.45
ALPHA_MAX = 0.80
BASE_ALPHA = 0.80
THRESHOLD = 10.0
GAIN = 0.25


# ============================================================
# Utility functions copied in statistical spirit from the
# frozen holdout50 analysis
# ============================================================

def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")

    # Rows may intentionally contain different fields
    # (e.g., task-score statistics versus success-rate
    # statistics). Build a stable union of all column names.
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    x = np.asarray(values, dtype=float)

    if x.size < 2:
        raise RuntimeError("Too few values")

    if not np.all(np.isfinite(x)):
        raise RuntimeError("Non-finite values")

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)

    order = np.argsort(p)
    adjusted = np.empty_like(p)

    running = 0.0
    m = len(p)

    for rank, index in enumerate(order):
        running = max(
            running,
            (m - rank) * p[index],
        )
        adjusted[index] = min(1.0, running)

    return adjusted


def safe_wilcoxon(diff):
    d = np.asarray(diff, dtype=float)

    if np.all(np.abs(d) <= TIE_TOL):
        return 0.0, 1.0

    try:
        r = stats.wilcoxon(
            d,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        r = stats.wilcoxon(
            d,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )

    return float(r.statistic), float(r.pvalue)


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    nz = d[np.abs(d) > TIE_TOL]

    if nz.size == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(nz),
        method="average",
    )

    pos = float(np.sum(ranks[nz > 0]))
    neg = float(np.sum(ranks[nz < 0]))

    return (pos - neg) / (pos + neg)


def exact_mcnemar(aasf_success, original_success):
    a = np.asarray(aasf_success, dtype=bool)
    o = np.asarray(original_success, dtype=bool)

    aasf_only = int(np.sum(a & ~o))
    original_only = int(np.sum(~a & o))

    discordant = aasf_only + original_only

    if discordant == 0:
        return 1.0, aasf_only, original_only

    try:
        p = stats.binomtest(
            min(aasf_only, original_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    except AttributeError:
        p = stats.binom_test(
            min(aasf_only, original_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        )

    return float(p), aasf_only, original_only


def bootstrap_mean_ci(values, seed):
    x = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    estimates = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    chunk = 5000

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk,
    ):
        stop = min(
            start + chunk,
            N_BOOTSTRAP,
        )

        idx = rng.integers(
            0,
            len(x),
            size=(stop - start, len(x)),
        )

        estimates[start:stop] = np.mean(
            x[idx],
            axis=1,
        )

    low, high = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return float(low), float(high)


def bootstrap_reduction_ci(original, aasf, seed):
    o = np.asarray(original, dtype=float)
    a = np.asarray(aasf, dtype=float)

    rng = np.random.default_rng(seed)

    absolute = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    relative = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    chunk = 5000

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk,
    ):
        stop = min(
            start + chunk,
            N_BOOTSTRAP,
        )

        idx = rng.integers(
            0,
            len(o),
            size=(stop - start, len(o)),
        )

        o_mean = np.mean(
            o[idx],
            axis=1,
        )

        a_mean = np.mean(
            a[idx],
            axis=1,
        )

        absolute[start:stop] = (
            o_mean - a_mean
        )

        relative[start:stop] = (
            100.0
            * (o_mean - a_mean)
            / o_mean
        )

    abs_ci = np.percentile(
        absolute,
        [2.5, 97.5],
    )

    rel_ci = np.percentile(
        relative,
        [2.5, 97.5],
    )

    return (
        (float(abs_ci[0]), float(abs_ci[1])),
        (float(rel_ci[0]), float(rel_ci[1])),
    )


def d_metric(actions, order):
    a = np.asarray(
        actions,
        dtype=float,
    )

    if (
        a.ndim != 2
        or a.shape[0] <= order
    ):
        raise RuntimeError(
            f"Invalid action shape "
            f"{a.shape} for D{order}"
        )

    diff = np.diff(
        a,
        n=order,
        axis=0,
    )

    return float(
        np.mean(
            np.linalg.norm(
                diff,
                axis=1,
            )
        )
    )


# ============================================================
# Load task scores
# ============================================================

scores = {}

for method in METHODS:

    p = RESULT_ROOT / method / "eval_log.json"

    with p.open("r") as f:
        log = json.load(f)

    vals = []

    for seed in SEEDS:

        key = f"test/sim_max_reward_{seed}"

        if key not in log:
            raise RuntimeError(
                f"Missing {key} in {p}"
            )

        vals.append(
            float(log[key])
        )

    scores[method] = np.asarray(
        vals,
        dtype=float,
    )


# ============================================================
# Load and audit traces
# ============================================================

traces = {
    method: {}
    for method in METHODS
}

mechanism_d = []
mechanism_alpha = []

first_action_max_diff = 0.0
reconstruction_max_diff = 0.0
original_identity_max_diff = 0.0
execution_accounting_max_diff = 0.0
initial_crn_max_diff = 0.0

for seed in SEEDS:

    seed_data = {}

    for method in METHODS:

        path = (
            TRACE_ROOT
            / method
            / f"seed_{seed}_{method}.npz"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing trace {path}"
            )

        with np.load(
            path,
            allow_pickle=True,
        ) as d:

            raw = np.asarray(
                d["raw_action"]
            )

            planned = np.asarray(
                d["executed_action"]
            )

            actual = np.asarray(
                d["actual_executed_action"]
            )

            executed_per_query = np.asarray(
                d["executed_steps_per_query"],
                dtype=np.int64,
            )

            actual_steps = int(
                np.asarray(
                    d["actual_executed_steps"]
                )
            )

            if bool(
                np.asarray(
                    d["bsop_enabled"]
                ).item()
            ):
                raise RuntimeError(
                    f"{path}: BSOP unexpectedly enabled"
                )

            if actual_steps != len(actual):
                raise RuntimeError(
                    f"{path}: execution count mismatch"
                )

            parts = []

            for q, n in enumerate(
                executed_per_query
            ):
                if n > 0:
                    parts.append(
                        planned[q, :n]
                    )

            rebuilt = np.concatenate(
                parts,
                axis=0,
            )

            if rebuilt.shape != actual.shape:
                raise RuntimeError(
                    f"{path}: rebuilt shape mismatch"
                )

            exec_diff = float(
                np.max(
                    np.abs(
                        rebuilt.astype(float)
                        - actual.astype(float)
                    )
                )
            )

            execution_accounting_max_diff = max(
                execution_accounting_max_diff,
                exec_diff,
            )

            if exec_diff > 1e-7:
                raise RuntimeError(
                    f"{path}: execution audit failed"
                )

            if method == "original":

                identity_diff = float(
                    np.max(
                        np.abs(
                            raw.astype(float)
                            - planned.astype(float)
                        )
                    )
                )

                original_identity_max_diff = max(
                    original_identity_max_diff,
                    identity_diff,
                )

                if identity_diff > 1e-7:
                    raise RuntimeError(
                        f"{path}: original action modified"
                    )

            else:

                first_diff = float(
                    np.max(
                        np.abs(
                            raw[:, 0].astype(float)
                            - planned[:, 0].astype(float)
                        )
                    )
                )

                first_action_max_diff = max(
                    first_action_max_diff,
                    first_diff,
                )

                if first_diff > 1e-7:
                    raise RuntimeError(
                        f"{path}: AASF first action changed"
                    )

                # Exact frozen AASF reconstruction.
                rec = raw.copy()

                q_d = []
                q_alpha = []

                for q in range(
                    raw.shape[0]
                ):

                    local_d = [np.nan]
                    local_alpha = [np.nan]

                    for t in range(
                        1,
                        raw.shape[1],
                    ):

                        delta = np.linalg.norm(
                            raw[q:q+1, t]
                            - rec[q:q+1, t-1],
                            axis=-1,
                            keepdims=True,
                        )

                        excess = np.maximum(
                            delta - THRESHOLD,
                            0.0,
                        ) / (
                            THRESHOLD + 1e-6
                        )

                        alpha = (
                            BASE_ALPHA
                            - GAIN * excess
                        )

                        alpha = np.clip(
                            alpha,
                            ALPHA_MIN,
                            ALPHA_MAX,
                        )

                        rec[q:q+1, t] = (
                            alpha
                            * raw[q:q+1, t]
                            + (
                                1.0 - alpha
                            )
                            * rec[q:q+1, t-1]
                        )

                        local_d.append(
                            float(delta[0, 0])
                        )

                        local_alpha.append(
                            float(alpha[0, 0])
                        )

                    q_d.append(local_d)
                    q_alpha.append(local_alpha)

                recon_diff = float(
                    np.max(
                        np.abs(
                            rec.astype(float)
                            - planned.astype(float)
                        )
                    )
                )

                reconstruction_max_diff = max(
                    reconstruction_max_diff,
                    recon_diff,
                )

                if recon_diff > 1e-6:
                    raise RuntimeError(
                        f"{path}: AASF reconstruction failed"
                    )

                # Adaptive diagnostics only for actions that
                # were actually executed.
                for q, n_exec in enumerate(
                    executed_per_query
                ):

                    for t in range(
                        1,
                        min(
                            int(n_exec),
                            raw.shape[1],
                        ),
                    ):

                        mechanism_d.append(
                            q_d[q][t]
                        )

                        mechanism_alpha.append(
                            q_alpha[q][t]
                        )

            seed_data[method] = {
                "actual": actual.astype(float),
                "raw": raw,
                "planned": planned,
                "executed_per_query":
                    executed_per_query,
            }

    # CRN audit: first policy call must match.
    crn_diff = float(
        np.max(
            np.abs(
                seed_data["original"]["raw"][0]
                .astype(float)
                - seed_data["aasf"]["raw"][0]
                .astype(float)
            )
        )
    )

    initial_crn_max_diff = max(
        initial_crn_max_diff,
        crn_diff,
    )

    traces["original"][seed] = (
        seed_data["original"]["actual"]
    )

    traces["aasf"][seed] = (
        seed_data["aasf"]["actual"]
    )


# ============================================================
# Episode-level D1/D2/D3
# ============================================================

full_values = {
    metric: {
        method: np.empty(
            len(SEEDS),
            dtype=float,
        )
        for method in METHODS
    }
    for metric in METRICS
}

common_values = {
    metric: {
        method: np.empty(
            len(SEEDS),
            dtype=float,
        )
        for method in METHODS
    }
    for metric in METRICS
}

episode_rows = []
unequal_length_count = 0
common_lengths = []

for i, seed in enumerate(SEEDS):

    original = traces["original"][seed]
    aasf = traces["aasf"][seed]

    len_o = len(original)
    len_a = len(aasf)

    if len_o != len_a:
        unequal_length_count += 1

    common_len = min(
        len_o,
        len_a,
    )

    if common_len <= 3:
        raise RuntimeError(
            f"Seed {seed}: common prefix too short"
        )

    common_lengths.append(common_len)

    row = {
        "seed": seed,
        "original_score":
            float(scores["original"][i]),
        "aasf_score":
            float(scores["aasf"][i]),
        "original_steps": len_o,
        "aasf_steps": len_a,
        "common_prefix_steps": common_len,
    }

    for metric, order in METRICS.items():

        o_full = d_metric(
            original,
            order,
        )

        a_full = d_metric(
            aasf,
            order,
        )

        o_common = d_metric(
            original[:common_len],
            order,
        )

        a_common = d_metric(
            aasf[:common_len],
            order,
        )

        full_values[metric]["original"][i] = (
            o_full
        )

        full_values[metric]["aasf"][i] = (
            a_full
        )

        common_values[metric]["original"][i] = (
            o_common
        )

        common_values[metric]["aasf"][i] = (
            a_common
        )

        row[f"original_{metric.lower()}_full"] = (
            o_full
        )

        row[f"aasf_{metric.lower()}_full"] = (
            a_full
        )

        row[f"original_{metric.lower()}_common"] = (
            o_common
        )

        row[f"aasf_{metric.lower()}_common"] = (
            a_common
        )

    episode_rows.append(row)

write_csv(
    EPISODE_CSV,
    episode_rows,
)


# ============================================================
# Task statistics
# ============================================================

orig_score = scores["original"]
aasf_score = scores["aasf"]

score_diff = aasf_score - orig_score

score_w, score_p = safe_wilcoxon(
    score_diff
)

score_ci = bootstrap_mean_ci(
    score_diff,
    BOOTSTRAP_SEED,
)

score_sd_diff = float(
    np.std(
        score_diff,
        ddof=1,
    )
)

task_rows = [{
    "outcome": "task_score",
    "original_mean":
        float(np.mean(orig_score)),
    "aasf_mean":
        float(np.mean(aasf_score)),
    "difference_aasf_minus_original":
        float(np.mean(score_diff)),
    "bootstrap_95_ci_low":
        score_ci[0],
    "bootstrap_95_ci_high":
        score_ci[1],
    "wins":
        int(np.sum(score_diff > TIE_TOL)),
    "ties":
        int(np.sum(
            np.abs(score_diff) <= TIE_TOL
        )),
    "losses":
        int(np.sum(score_diff < -TIE_TOL)),
    "wilcoxon_statistic":
        score_w,
    "wilcoxon_p":
        score_p,
    "rank_biserial":
        rank_biserial(score_diff),
    "cohen_dz":
        (
            float(np.mean(score_diff))
            / score_sd_diff
            if score_sd_diff > 0
            else 0.0
        ),
}]

sr_rows = []

for idx, threshold in enumerate(
    SUCCESS_THRESHOLDS
):

    orig_success = (
        orig_score >= threshold
    )

    aasf_success = (
        aasf_score >= threshold
    )

    sr_diff = (
        aasf_success.astype(float)
        - orig_success.astype(float)
    )

    sr_ci = bootstrap_mean_ci(
        sr_diff,
        BOOTSTRAP_SEED + 100 + idx,
    )

    mcnemar_p, aasf_only, original_only = (
        exact_mcnemar(
            aasf_success,
            orig_success,
        )
    )

    sr_rows.append({
        "outcome": f"SR@{threshold:.2f}",
        "threshold": threshold,
        "original_success_count":
            int(np.sum(orig_success)),
        "original_success_rate":
            float(np.mean(orig_success)),
        "aasf_success_count":
            int(np.sum(aasf_success)),
        "aasf_success_rate":
            float(np.mean(aasf_success)),
        "difference_aasf_minus_original":
            float(np.mean(sr_diff)),
        "bootstrap_95_ci_low":
            sr_ci[0],
        "bootstrap_95_ci_high":
            sr_ci[1],
        "aasf_only_successes":
            aasf_only,
        "original_only_successes":
            original_only,
        "mcnemar_p_raw":
            mcnemar_p,
    })

sr_holm = holm_adjust(
    [r["mcnemar_p_raw"] for r in sr_rows]
)

for row, p_adj in zip(
    sr_rows,
    sr_holm,
):
    row["mcnemar_p_holm_across_2_thresholds"] = (
        float(p_adj)
    )

task_rows.extend(sr_rows)

write_csv(
    TASK_CSV,
    task_rows,
)


# ============================================================
# Smoothness statistics
# ============================================================

scope_values = {
    "common_prefix": common_values,
    "full_trace": full_values,
}

smooth_rows = []

for scope_index, scope in enumerate(SCOPES):

    scope_rows = []

    for metric_index, metric in enumerate(
        METRICS
    ):

        o = scope_values[scope][metric][
            "original"
        ]

        a = scope_values[scope][metric][
            "aasf"
        ]

        reduction = o - a

        w, p = safe_wilcoxon(
            reduction
        )

        abs_ci, rel_ci = (
            bootstrap_reduction_ci(
                o,
                a,
                BOOTSTRAP_SEED
                + 1000
                + scope_index * 100
                + metric_index,
            )
        )

        sd_diff = float(
            np.std(
                reduction,
                ddof=1,
            )
        )

        scope_rows.append({
            "analysis_scope": scope,
            "metric": metric,
            "n": len(SEEDS),

            "original_mean":
                float(np.mean(o)),
            "original_sd":
                float(np.std(o, ddof=1)),

            "aasf_mean":
                float(np.mean(a)),
            "aasf_sd":
                float(np.std(a, ddof=1)),

            "mean_absolute_reduction":
                float(np.mean(reduction)),

            "relative_mean_reduction_percent":
                float(
                    100.0
                    * (
                        np.mean(o)
                        - np.mean(a)
                    )
                    / np.mean(o)
                ),

            "absolute_bootstrap_95_ci_low":
                abs_ci[0],
            "absolute_bootstrap_95_ci_high":
                abs_ci[1],

            "relative_bootstrap_95_ci_low_percent":
                rel_ci[0],
            "relative_bootstrap_95_ci_high_percent":
                rel_ci[1],

            "aasf_smoother_wins":
                int(
                    np.sum(
                        reduction > TIE_TOL
                    )
                ),

            "ties":
                int(
                    np.sum(
                        np.abs(reduction)
                        <= TIE_TOL
                    )
                ),

            "aasf_rougher_losses":
                int(
                    np.sum(
                        reduction < -TIE_TOL
                    )
                ),

            "wilcoxon_statistic": w,
            "wilcoxon_p_raw": p,

            "rank_biserial":
                rank_biserial(reduction),

            "cohen_dz":
                (
                    float(np.mean(reduction))
                    / sd_diff
                    if sd_diff > 0
                    else 0.0
                ),
        })

    adjusted = holm_adjust(
        [
            row["wilcoxon_p_raw"]
            for row in scope_rows
        ]
    )

    for row, p_adj in zip(
        scope_rows,
        adjusted,
    ):
        row[
            "wilcoxon_p_holm_across_d1_d2_d3"
        ] = float(p_adj)

    smooth_rows.extend(scope_rows)

write_csv(
    SMOOTH_CSV,
    smooth_rows,
)


# ============================================================
# AASF mechanism statistics
# ============================================================

mechanism_d = np.asarray(
    mechanism_d,
    dtype=float,
)

mechanism_alpha = np.asarray(
    mechanism_alpha,
    dtype=float,
)

mechanism_rows = [{
    "n_executed_adaptive_decisions":
        int(len(mechanism_d)),

    "d_mean":
        float(np.mean(mechanism_d)),
    "d_sd":
        float(np.std(
            mechanism_d,
            ddof=1,
        )),
    "d_median":
        float(np.median(mechanism_d)),
    "d_q1":
        float(np.percentile(
            mechanism_d,
            25,
        )),
    "d_q3":
        float(np.percentile(
            mechanism_d,
            75,
        )),
    "d_p90":
        float(np.percentile(
            mechanism_d,
            90,
        )),
    "d_p95":
        float(np.percentile(
            mechanism_d,
            95,
        )),

    "d_gt_tau_percent":
        float(
            100.0
            * np.mean(
                mechanism_d > THRESHOLD
            )
        ),

    "alpha_mean":
        float(np.mean(
            mechanism_alpha
        )),
    "alpha_sd":
        float(np.std(
            mechanism_alpha,
            ddof=1,
        )),
    "alpha_median":
        float(np.median(
            mechanism_alpha
        )),

    "alpha_at_min_percent":
        float(
            100.0
            * np.mean(
                np.isclose(
                    mechanism_alpha,
                    ALPHA_MIN,
                )
            )
        ),

    "alpha_at_max_percent":
        float(
            100.0
            * np.mean(
                np.isclose(
                    mechanism_alpha,
                    ALPHA_MAX,
                )
            )
        ),

    "first_action_max_abs_diff":
        first_action_max_diff,

    "aasf_reconstruction_max_abs_diff":
        reconstruction_max_diff,

    "original_identity_max_abs_diff":
        original_identity_max_diff,

    "execution_accounting_max_abs_diff":
        execution_accounting_max_diff,

    "initial_crn_first_chunk_max_abs_diff":
        initial_crn_max_diff,
}]

write_csv(
    MECH_CSV,
    mechanism_rows,
)


# ============================================================
# JSON + human-readable report
# ============================================================

result = {
    "protocol": {
        "run": RUN,
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "n": len(SEEDS),
        "bootstrap_resamples":
            N_BOOTSTRAP,
        "bootstrap_seed":
            BOOTSTRAP_SEED,
        "image_specific_retuning":
            False,
        "aasf_parameters": {
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "tau": THRESHOLD,
            "gamma": GAIN,
        },
    },

    "rollout_lengths": {
        "unequal_length_pairs":
            unequal_length_count,
        "common_prefix_steps":
            summarize(common_lengths),
    },

    "task_statistics":
        task_rows,

    "smoothness_statistics":
        smooth_rows,

    "mechanism_statistics":
        mechanism_rows[0],
}

JSON_OUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)

lines = []

lines.append(
    "IMAGE PUSHT AASF CONFIRMATORY FRESH-50"
)
lines.append("=" * 72)

lines.append("")
lines.append("PROTOCOL")
lines.append(
    f"Seeds: {SEEDS[0]}-{SEEDS[-1]} "
    f"(n={len(SEEDS)})"
)
lines.append(
    "Image-specific AASF retuning: NONE"
)
lines.append(
    "AASF: alpha_min=0.45, "
    "alpha_max=0.80, tau=10.0, gamma=0.25"
)

lines.append("")
lines.append("TASK PERFORMANCE")

lines.append(
    f"Original: "
    f"{np.mean(orig_score):.6f} "
    f"+/- {np.std(orig_score, ddof=1):.6f}"
)

lines.append(
    f"AASF:     "
    f"{np.mean(aasf_score):.6f} "
    f"+/- {np.std(aasf_score, ddof=1):.6f}"
)

lines.append(
    f"AASF - Original = "
    f"{np.mean(score_diff):+.6f}, "
    f"bootstrap 95% CI="
    f"[{score_ci[0]:+.6f}, "
    f"{score_ci[1]:+.6f}]"
)

lines.append(
    f"Wilcoxon p="
    f"{score_p:.10g}, "
    f"rank-biserial="
    f"{rank_biserial(score_diff):+.4f}"
)

for row in sr_rows:

    lines.append(
        f"{row['outcome']}: "
        f"Original="
        f"{row['original_success_rate']:.1%}, "
        f"AASF="
        f"{row['aasf_success_rate']:.1%}, "
        f"diff="
        f"{row['difference_aasf_minus_original']:+.1%}, "
        f"95% CI=["
        f"{row['bootstrap_95_ci_low']:+.1%}, "
        f"{row['bootstrap_95_ci_high']:+.1%}], "
        f"McNemar raw p="
        f"{row['mcnemar_p_raw']:.10g}, "
        f"Holm p="
        f"{row['mcnemar_p_holm_across_2_thresholds']:.10g}"
    )

lines.append("")
lines.append(
    "ROLLOUT LENGTH / COMMON PREFIX"
)

lines.append(
    f"Unequal rollout lengths: "
    f"{unequal_length_count}/50"
)

lines.append(
    f"Common-prefix steps: "
    f"mean={np.mean(common_lengths):.2f}, "
    f"median={np.median(common_lengths):.2f}, "
    f"min={np.min(common_lengths)}, "
    f"max={np.max(common_lengths)}"
)

for scope in SCOPES:

    lines.append("")
    lines.append(
        f"SMOOTHNESS: {scope.upper()}"
    )

    for row in smooth_rows:

        if row["analysis_scope"] != scope:
            continue

        lines.append(
            f"{row['metric']}: "
            f"Original="
            f"{row['original_mean']:.6f}, "
            f"AASF="
            f"{row['aasf_mean']:.6f}, "
            f"reduction="
            f"{row['relative_mean_reduction_percent']:+.2f}%, "
            f"relative bootstrap 95% CI=["
            f"{row['relative_bootstrap_95_ci_low_percent']:+.2f}%, "
            f"{row['relative_bootstrap_95_ci_high_percent']:+.2f}%], "
            f"smoother="
            f"{row['aasf_smoother_wins']}/50, "
            f"rougher="
            f"{row['aasf_rougher_losses']}/50, "
            f"Wilcoxon raw p="
            f"{row['wilcoxon_p_raw']:.10g}, "
            f"Holm p="
            f"{row['wilcoxon_p_holm_across_d1_d2_d3']:.10g}, "
            f"rrb="
            f"{row['rank_biserial']:+.4f}"
        )

lines.append("")
lines.append(
    "AASF MECHANISM AUDIT "
    "(ACTUALLY EXECUTED DECISIONS)"
)

m = mechanism_rows[0]

lines.append(
    f"Adaptive decisions: "
    f"{m['n_executed_adaptive_decisions']}"
)

lines.append(
    f"d median="
    f"{m['d_median']:.4f}, "
    f"Q1/Q3="
    f"{m['d_q1']:.4f}/"
    f"{m['d_q3']:.4f}, "
    f"P90/P95="
    f"{m['d_p90']:.4f}/"
    f"{m['d_p95']:.4f}"
)

lines.append(
    f"d > tau: "
    f"{m['d_gt_tau_percent']:.2f}%"
)

lines.append(
    f"alpha mean +/- SD: "
    f"{m['alpha_mean']:.4f} +/- "
    f"{m['alpha_sd']:.4f}"
)

lines.append(
    f"alpha=min: "
    f"{m['alpha_at_min_percent']:.2f}%"
)

lines.append(
    f"alpha=max: "
    f"{m['alpha_at_max_percent']:.2f}%"
)

lines.append(
    f"First-action max abs diff: "
    f"{m['first_action_max_abs_diff']:.9g}"
)

lines.append(
    f"AASF reconstruction max abs diff: "
    f"{m['aasf_reconstruction_max_abs_diff']:.9g}"
)

lines.append(
    f"Original identity max abs diff: "
    f"{m['original_identity_max_abs_diff']:.9g}"
)

lines.append(
    f"Execution accounting max abs diff: "
    f"{m['execution_accounting_max_abs_diff']:.9g}"
)

lines.append(
    f"Initial CRN first-chunk max abs diff: "
    f"{m['initial_crn_first_chunk_max_abs_diff']:.9g}"
)

REPORT_OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

print()
print("OUTPUT FILES")
print(EPISODE_CSV)
print(TASK_CSV)
print(SMOOTH_CSV)
print(MECH_CSV)
print(JSON_OUT)
print(REPORT_OUT)
