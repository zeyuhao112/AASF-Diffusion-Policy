import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

SEEDS = list(range(4300050, 4300100))
SEED_SET = set(SEEDS)
SUCCESS_THRESHOLD = 0.90
N_BOOTSTRAP = 100000
BOOTSTRAP_SEED = 20260714
TIE_TOL = 1e-12

METHODS = {
    "Original": "crn_final_original_100",
    "SMA-3": "crn_final_sma3_100",
    "SMA-5": "crn_final_sma5_100",
    "Fixed EMA (alpha=0.7)": "crn_final_fixed_ema_a07_100",
    "AASF": "crn_final_aasf_100",
}
BASELINES = [
    "Original",
    "SMA-3",
    "SMA-5",
    "Fixed EMA (alpha=0.7)",
]
METRICS = {"D1": 1, "D2": 2, "D3": 3}
SCOPES = ("full_trace", "common_prefix")

TRACE_ROOT = Path("paper_artifacts/traces")
TABLE_DIR = Path("paper_exp/tables")
PREFIX = "holdout50"

TASK_SUMMARY_CSV = TABLE_DIR / f"{PREFIX}_five_method_task_summary.csv"
EPISODE_CSV = TABLE_DIR / f"{PREFIX}_five_method_episode_metrics.csv"
TASK_PAIRED_CSV = TABLE_DIR / f"{PREFIX}_aasf_vs_baselines_task_paired.csv"
SMOOTH_SUMMARY_CSV = TABLE_DIR / f"{PREFIX}_five_method_smoothness_summary.csv"
SMOOTH_FRIEDMAN_CSV = TABLE_DIR / f"{PREFIX}_five_method_smoothness_friedman.csv"
SMOOTH_PAIRED_CSV = TABLE_DIR / f"{PREFIX}_aasf_vs_baselines_smoothness_paired.csv"
JSON_OUTPUT = TABLE_DIR / f"{PREFIX}_five_method_statistics.json"
TEXT_OUTPUT = TABLE_DIR / f"{PREFIX}_five_method_report.txt"
LATEX_OUTPUT = TABLE_DIR / f"{PREFIX}_paper_tables.tex"


def method_key(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "")
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    x = np.asarray(values, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise RuntimeError("Invalid values for summary")
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
    m = p.size
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def safe_wilcoxon(diff):
    d = np.asarray(diff, dtype=float)
    if np.all(np.abs(d) <= TIE_TOL):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(
            d,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        result = stats.wilcoxon(
            d,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )
    return float(result.statistic), float(result.pvalue)


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    nz = d[np.abs(d) > TIE_TOL]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz), method="average")
    pos = float(np.sum(ranks[nz > 0]))
    neg = float(np.sum(ranks[nz < 0]))
    return (pos - neg) / (pos + neg)


def exact_mcnemar_p(aasf_success, baseline_success):
    a = np.asarray(aasf_success, dtype=bool)
    b = np.asarray(baseline_success, dtype=bool)
    aasf_only = int(np.sum(a & ~b))
    baseline_only = int(np.sum(~a & b))
    discordant = aasf_only + baseline_only
    if discordant == 0:
        return 1.0, aasf_only, baseline_only
    try:
        p = stats.binomtest(
            min(aasf_only, baseline_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    except AttributeError:
        p = stats.binom_test(
            min(aasf_only, baseline_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        )
    return float(p), aasf_only, baseline_only


def bootstrap_mean_ci(values, seed):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=float)
    chunk = 5000
    for start in range(0, N_BOOTSTRAP, chunk):
        stop = min(start + chunk, N_BOOTSTRAP)
        idx = rng.integers(0, x.size, size=(stop - start, x.size))
        estimates[start:stop] = np.mean(x[idx], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_reduction_ci(baseline, aasf, seed):
    b = np.asarray(baseline, dtype=float)
    a = np.asarray(aasf, dtype=float)
    rng = np.random.default_rng(seed)
    absolute = np.empty(N_BOOTSTRAP, dtype=float)
    relative = np.empty(N_BOOTSTRAP, dtype=float)
    chunk = 5000
    for start in range(0, N_BOOTSTRAP, chunk):
        stop = min(start + chunk, N_BOOTSTRAP)
        idx = rng.integers(0, b.size, size=(stop - start, b.size))
        b_mean = np.mean(b[idx], axis=1)
        a_mean = np.mean(a[idx], axis=1)
        absolute[start:stop] = b_mean - a_mean
        relative[start:stop] = 100.0 * (b_mean - a_mean) / b_mean
    abs_ci = np.percentile(absolute, [2.5, 97.5])
    rel_ci = np.percentile(relative, [2.5, 97.5])
    return (
        (float(abs_ci[0]), float(abs_ci[1])),
        (float(rel_ci[0]), float(rel_ci[1])),
    )


def d_metric(actions, order):
    a = np.asarray(actions, dtype=float)
    if a.ndim != 2 or a.shape[0] <= order:
        raise RuntimeError(f"Invalid action shape {a.shape} for D{order}")
    diff = np.diff(a, n=order, axis=0)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def load_method(dirname):
    directory = TRACE_ROOT / dirname
    records = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            required = {"seed", "score", "filtered_action", "chunk_starts"}
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(f"{path}: missing {sorted(missing)}")
            seed = int(np.asarray(data["seed"]).item())
            if seed not in SEED_SET:
                continue
            score = float(np.asarray(data["score"]).item())
            actions = np.asarray(data["filtered_action"], dtype=float)
            chunk_starts = np.asarray(data["chunk_starts"])
        if seed in records:
            raise RuntimeError(f"{directory}: duplicate seed {seed}")
        if actions.ndim != 2 or actions.shape[0] <= 3:
            raise RuntimeError(f"{path}: invalid action shape {actions.shape}")
        if chunk_starts.ndim != 1:
            raise RuntimeError(f"{path}: invalid chunk shape {chunk_starts.shape}")
        if not np.isfinite(score) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"{path}: non-finite values")
        records[seed] = {
            "path": str(path),
            "score": score,
            "actions": actions,
            "chunk_starts": chunk_starts,
        }
    if set(records) != SEED_SET:
        missing = sorted(SEED_SET - set(records))
        extra = sorted(set(records) - SEED_SET)
        raise RuntimeError(
            f"{directory}: holdout seed mismatch; missing={missing}, extra={extra}"
        )
    return records


TABLE_DIR.mkdir(parents=True, exist_ok=True)
traces = {method: load_method(dirname) for method, dirname in METHODS.items()}

scores = {
    method: np.asarray([traces[method][seed]["score"] for seed in SEEDS])
    for method in METHODS
}

full_values = {
    metric: {method: np.empty(len(SEEDS), dtype=float) for method in METHODS}
    for metric in METRICS
}
common_values = {
    metric: {method: np.empty(len(SEEDS), dtype=float) for method in METHODS}
    for metric in METRICS
}

episode_rows = []
unequal_length_count = 0
for i, seed in enumerate(SEEDS):
    lengths = {
        method: int(traces[method][seed]["actions"].shape[0])
        for method in METHODS
    }
    if len(set(lengths.values())) > 1:
        unequal_length_count += 1
    common_len = min(lengths.values())
    row = {"seed": seed, "common_prefix_steps": common_len}
    for method in METHODS:
        key = method_key(method)
        actions = traces[method][seed]["actions"]
        row[f"{key}_score"] = float(scores[method][i])
        row[f"{key}_action_steps"] = lengths[method]
        for metric, order in METRICS.items():
            full = d_metric(actions, order)
            common = d_metric(actions[:common_len], order)
            full_values[metric][method][i] = full
            common_values[metric][method][i] = common
            row[f"{key}_{metric.lower()}_full"] = full
            row[f"{key}_{metric.lower()}_common"] = common
    episode_rows.append(row)
write_csv(EPISODE_CSV, episode_rows)

# Task summaries and omnibus test.
task_summary_rows = []
for method in METHODS:
    s = summarize(scores[method])
    success = scores[method] >= SUCCESS_THRESHOLD
    task_summary_rows.append({
        "method": method,
        **s,
        "success_threshold": SUCCESS_THRESHOLD,
        "success_count": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
    })
write_csv(TASK_SUMMARY_CSV, task_summary_rows)

task_friedman = stats.friedmanchisquare(*[scores[m] for m in METHODS])
task_kendall_w = float(task_friedman.statistic / (len(SEEDS) * (len(METHODS) - 1)))

# Planned task comparisons: AASF versus four baselines.
task_rows = []
aasf_score = scores["AASF"]
for index, baseline in enumerate(BASELINES):
    base = scores[baseline]
    diff = aasf_score - base  # Positive means higher AASF score.
    wilcoxon_stat, wilcoxon_p = safe_wilcoxon(diff)
    ci_low, ci_high = bootstrap_mean_ci(diff, BOOTSTRAP_SEED + index)
    aasf_success = aasf_score >= SUCCESS_THRESHOLD
    base_success = base >= SUCCESS_THRESHOLD
    mcnemar_p, aasf_only, baseline_only = exact_mcnemar_p(
        aasf_success, base_success
    )
    sr_diff = aasf_success.astype(float) - base_success.astype(float)
    sr_ci_low, sr_ci_high = bootstrap_mean_ci(
        sr_diff, BOOTSTRAP_SEED + 100 + index
    )
    wins = int(np.sum(diff > TIE_TOL))
    losses = int(np.sum(diff < -TIE_TOL))
    ties = int(diff.size - wins - losses)
    sd_diff = float(np.std(diff, ddof=1))
    task_rows.append({
        "baseline": baseline,
        "n": len(SEEDS),
        "aasf_mean": float(np.mean(aasf_score)),
        "baseline_mean": float(np.mean(base)),
        "mean_score_difference_aasf_minus_baseline": float(np.mean(diff)),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_raw": wilcoxon_p,
        "rank_biserial": rank_biserial(diff),
        "cohen_dz": float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0,
        "aasf_success_count": int(np.sum(aasf_success)),
        "baseline_success_count": int(np.sum(base_success)),
        "success_rate_difference": float(np.mean(sr_diff)),
        "success_rate_bootstrap_95_ci_low": sr_ci_low,
        "success_rate_bootstrap_95_ci_high": sr_ci_high,
        "aasf_only_successes": aasf_only,
        "baseline_only_successes": baseline_only,
        "mcnemar_p_raw": mcnemar_p,
    })

wilcoxon_holm = holm_adjust([row["wilcoxon_p_raw"] for row in task_rows])
mcnemar_holm = holm_adjust([row["mcnemar_p_raw"] for row in task_rows])
for row, wp, mp in zip(task_rows, wilcoxon_holm, mcnemar_holm):
    row["wilcoxon_p_holm_across_4"] = float(wp)
    row["mcnemar_p_holm_across_4"] = float(mp)
write_csv(TASK_PAIRED_CSV, task_rows)

# Smoothness summaries, omnibus tests, and planned comparisons.
values_by_scope = {
    "full_trace": full_values,
    "common_prefix": common_values,
}
smooth_summary_rows = []
smooth_friedman_rows = []
smooth_paired_rows = []

for scope in SCOPES:
    values = values_by_scope[scope]
    scope_friedman = []
    for metric in METRICS:
        for method in METHODS:
            smooth_summary_rows.append({
                "analysis_scope": scope,
                "metric": metric,
                "method": method,
                **summarize(values[metric][method]),
            })
        result = stats.friedmanchisquare(*[values[metric][m] for m in METHODS])
        scope_friedman.append({
            "analysis_scope": scope,
            "metric": metric,
            "friedman_chi_square": float(result.statistic),
            "friedman_p_raw": float(result.pvalue),
            "kendalls_w": float(
                result.statistic / (len(SEEDS) * (len(METHODS) - 1))
            ),
        })
    friedman_holm = holm_adjust([r["friedman_p_raw"] for r in scope_friedman])
    for row, p_adj in zip(scope_friedman, friedman_holm):
        row["friedman_p_holm_across_d1_d2_d3"] = float(p_adj)
    smooth_friedman_rows.extend(scope_friedman)

    scope_pairs = []
    for metric_index, metric in enumerate(METRICS):
        aasf = values[metric]["AASF"]
        for base_index, baseline in enumerate(BASELINES):
            base = values[metric][baseline]
            reduction = base - aasf  # Positive means smoother AASF.
            wilcoxon_stat, wilcoxon_p = safe_wilcoxon(reduction)
            abs_ci, rel_ci = bootstrap_reduction_ci(
                base,
                aasf,
                BOOTSTRAP_SEED + 1000 + metric_index * 10 + base_index,
            )
            wins = int(np.sum(reduction > TIE_TOL))
            losses = int(np.sum(reduction < -TIE_TOL))
            ties = int(reduction.size - wins - losses)
            sd_diff = float(np.std(reduction, ddof=1))
            scope_pairs.append({
                "analysis_scope": scope,
                "metric": metric,
                "baseline": baseline,
                "n": len(SEEDS),
                "baseline_mean": float(np.mean(base)),
                "aasf_mean": float(np.mean(aasf)),
                "mean_absolute_reduction_baseline_minus_aasf": float(
                    np.mean(reduction)
                ),
                "relative_mean_reduction_percent": float(
                    100.0 * (np.mean(base) - np.mean(aasf)) / np.mean(base)
                ),
                "absolute_reduction_bootstrap_95_ci_low": abs_ci[0],
                "absolute_reduction_bootstrap_95_ci_high": abs_ci[1],
                "relative_reduction_bootstrap_95_ci_low_percent": rel_ci[0],
                "relative_reduction_bootstrap_95_ci_high_percent": rel_ci[1],
                "aasf_smoother_wins": wins,
                "ties": ties,
                "aasf_rougher_losses": losses,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_raw": wilcoxon_p,
                "rank_biserial": rank_biserial(reduction),
                "cohen_dz": float(np.mean(reduction) / sd_diff)
                if sd_diff > 0
                else 0.0,
            })
    pair_holm = holm_adjust([r["wilcoxon_p_raw"] for r in scope_pairs])
    for row, p_adj in zip(scope_pairs, pair_holm):
        row["wilcoxon_p_holm_across_12"] = float(p_adj)
    smooth_paired_rows.extend(scope_pairs)

write_csv(SMOOTH_SUMMARY_CSV, smooth_summary_rows)
write_csv(SMOOTH_FRIEDMAN_CSV, smooth_friedman_rows)
write_csv(SMOOTH_PAIRED_CSV, smooth_paired_rows)

# Compact LaTeX tables for manuscript insertion.
summary_lookup = {row["method"]: row for row in task_summary_rows}
smooth_lookup = {
    (row["analysis_scope"], row["metric"], row["method"]): row
    for row in smooth_summary_rows
}
latex = [
    r"% Holdout seeds 4300050--4300099; generated automatically.",
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Performance and action smoothness on the 50 held-out seeds.}",
    r"\label{tab:holdout50_main}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"Method & Score & SR@0.90 & D1 & D2 & D3 \\",
    r"\midrule",
]
for method in METHODS:
    task = summary_lookup[method]
    d1 = smooth_lookup[("full_trace", "D1", method)]
    d2 = smooth_lookup[("full_trace", "D2", method)]
    d3 = smooth_lookup[("full_trace", "D3", method)]
    label = method.replace("alpha=0.7", r"$\alpha=0.7$")
    latex.append(
        f"{label} & {task['mean']:.4f} $\\pm$ {task['sd']:.4f} & "
        f"{100.0 * task['success_rate']:.1f}\\% & "
        f"{d1['mean']:.4f} & {d2['mean']:.4f} & {d3['mean']:.4f} \\\\"
    )
latex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
    "",
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Planned paired task-performance comparisons on the held-out seeds.}",
    r"\label{tab:holdout50_task_pairs}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Baseline & $\Delta$Score & 95\% CI & Holm $p$ & $\Delta$SR \\",
    r"\midrule",
])
for row in task_rows:
    baseline = row["baseline"].replace("alpha=0.7", r"$\alpha=0.7$")
    latex.append(
        f"{baseline} & {row['mean_score_difference_aasf_minus_baseline']:+.4f} & "
        f"[{row['bootstrap_95_ci_low']:+.4f}, {row['bootstrap_95_ci_high']:+.4f}] & "
        f"{row['wilcoxon_p_holm_across_4']:.3g} & "
        f"{100.0 * row['success_rate_difference']:+.1f}\\% \\\\"
    )
latex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])
LATEX_OUTPUT.write_text("\n".join(latex) + "\n", encoding="utf-8")

# JSON provenance and all numeric results.
input_manifest = {}
for method, dirname in METHODS.items():
    files = [Path(traces[method][seed]["path"]) for seed in SEEDS]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    input_manifest[method] = {
        "directory": str(TRACE_ROOT / dirname),
        "holdout_trace_count": len(files),
        "holdout_manifest_sha256": digest.hexdigest(),
    }

combined = {
    "analysis": {
        "name": "50-seed held-out subset analysis",
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_paired_seeds": len(SEEDS),
        "success_threshold": SUCCESS_THRESHOLD,
        "metric_definition": "Dk = mean_t ||Delta^k a_t||_2; no time-step scaling",
        "primary_smoothness_scope": "full_trace",
        "sensitivity_smoothness_scope": "common_prefix",
        "unequal_action_length_episode_count": unequal_length_count,
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    },
    "input_manifest": input_manifest,
    "task_summary": task_summary_rows,
    "task_friedman": {
        "chi_square": float(task_friedman.statistic),
        "p_value": float(task_friedman.pvalue),
        "kendalls_w": task_kendall_w,
    },
    "task_aasf_vs_baselines": task_rows,
    "smoothness_summary": smooth_summary_rows,
    "smoothness_friedman": smooth_friedman_rows,
    "smoothness_aasf_vs_baselines": smooth_paired_rows,
}
JSON_OUTPUT.write_text(
    json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

# Human-readable report.
lines = [
    "==== Held-out 50-seed analysis (4300050-4300099) ====",
    "",
    "Task-performance summaries:",
]
for row in task_summary_rows:
    lines.append(
        f"  {row['method']}: mean={row['mean']:.8f} ± {row['sd']:.8f}, "
        f"median={row['median']:.8f}, success={row['success_count']}/50 "
        f"({row['success_rate']:.1%})"
    )
lines.extend([
    "",
    "Five-method task Friedman test:",
    f"  chi-square={task_friedman.statistic:.8f}, "
    f"p={task_friedman.pvalue:.10g}, Kendall's W={task_kendall_w:.6f}",
    "",
    "Planned task comparisons (positive score difference favors AASF):",
])
for row in task_rows:
    lines.extend([
        f"  AASF vs {row['baseline']}:",
        f"    Mean score difference={row['mean_score_difference_aasf_minus_baseline']:+.8f}, "
        f"bootstrap 95% CI=[{row['bootstrap_95_ci_low']:+.8f}, "
        f"{row['bootstrap_95_ci_high']:+.8f}]",
        f"    Win/tie/loss={row['wins']}/{row['ties']}/{row['losses']}",
        f"    Wilcoxon p(raw)={row['wilcoxon_p_raw']:.10g}, "
        f"p(Holm)={row['wilcoxon_p_holm_across_4']:.10g}, "
        f"rank-biserial={row['rank_biserial']:+.6f}",
        f"    Success-rate difference={row['success_rate_difference']:+.1%}, "
        f"bootstrap 95% CI=[{row['success_rate_bootstrap_95_ci_low']:+.1%}, "
        f"{row['success_rate_bootstrap_95_ci_high']:+.1%}]",
        f"    McNemar p(raw)={row['mcnemar_p_raw']:.10g}, "
        f"p(Holm)={row['mcnemar_p_holm_across_4']:.10g}",
    ])

for scope in SCOPES:
    lines.extend([
        "",
        "=" * 72,
        f"Smoothness scope: {scope}",
        "=" * 72,
        "Method summaries:",
    ])
    for metric in METRICS:
        lines.append(f"  {metric}:")
        for method in METHODS:
            row = smooth_lookup[(scope, metric, method)]
            lines.append(
                f"    {method}: {row['mean']:.8f} ± {row['sd']:.8f}, "
                f"median={row['median']:.8f}"
            )
    lines.append("Five-method Friedman tests:")
    for row in smooth_friedman_rows:
        if row["analysis_scope"] == scope:
            lines.append(
                f"  {row['metric']}: chi-square={row['friedman_chi_square']:.8f}, "
                f"p(raw)={row['friedman_p_raw']:.10g}, "
                f"p(Holm)={row['friedman_p_holm_across_d1_d2_d3']:.10g}, "
                f"Kendall's W={row['kendalls_w']:.6f}"
            )
    lines.append("Planned smoothness comparisons (positive reduction = smoother AASF):")
    for row in smooth_paired_rows:
        if row["analysis_scope"] != scope:
            continue
        lines.extend([
            f"  {row['metric']}: AASF vs {row['baseline']}",
            f"    Mean reduction={row['mean_absolute_reduction_baseline_minus_aasf']:+.8f} "
            f"({row['relative_mean_reduction_percent']:+.3f}%), "
            f"bootstrap 95% CI=[{row['absolute_reduction_bootstrap_95_ci_low']:+.8f}, "
            f"{row['absolute_reduction_bootstrap_95_ci_high']:+.8f}]",
            f"    Smoother/tie/rougher={row['aasf_smoother_wins']}/"
            f"{row['ties']}/{row['aasf_rougher_losses']}",
            f"    Wilcoxon p(raw)={row['wilcoxon_p_raw']:.10g}, "
            f"p(Holm across 12)={row['wilcoxon_p_holm_across_12']:.10g}, "
            f"rank-biserial={row['rank_biserial']:+.6f}",
        ])

lines.extend([
    "",
    f"Unequal action lengths across methods: {unequal_length_count}/50 episodes",
    "",
    "Saved files:",
    f"  {TASK_SUMMARY_CSV}",
    f"  {EPISODE_CSV}",
    f"  {TASK_PAIRED_CSV}",
    f"  {SMOOTH_SUMMARY_CSV}",
    f"  {SMOOTH_FRIEDMAN_CSV}",
    f"  {SMOOTH_PAIRED_CSV}",
    f"  {JSON_OUTPUT}",
    f"  {TEXT_OUTPUT}",
    f"  {LATEX_OUTPUT}",
])
report = "\n".join(lines) + "\n"
TEXT_OUTPUT.write_text(report, encoding="utf-8")
print(report, end="")
