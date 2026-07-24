#!/usr/bin/env python3
"""
Chunk-internal versus chunk-boundary smoothness analysis for AASF traces.

This script reads the frozen AASF NPZ traces only. It does not run the policy,
perform rollout, or require a GPU.
"""

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats


TRACE_RELATIVE = Path(
    "paper_artifacts/traces/crn_final_aasf_100"
)
OUTPUT_RELATIVE = Path(
    "paper_exp/tables/chunk_boundary_analysis"
)
EXPECTED_ALL = tuple(range(4300000, 4300100))
EXPECTED_HOLDOUT = tuple(range(4300050, 4300100))
BOOTSTRAP_SAMPLES = 100000
BASE_BOOTSTRAP_SEED = 20260714
FIRST_ACTION_ATOL = 1.0e-7
ZERO_TOL = 1.0e-12


def scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            "Expected a scalar value, got shape "
            f"{arr.shape}"
        )
    return arr.reshape(()).item()


def python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def json_ready(obj):
    if isinstance(obj, dict):
        return {
            str(key): json_ready(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [json_ready(value) for value in obj]
    return python_value(obj)


def load_traces(trace_dir):
    files = sorted(trace_dir.glob("*.npz"))
    if len(files) != 100:
        raise RuntimeError(
            f"{trace_dir}: expected 100 NPZ files, "
            f"found {len(files)}"
        )

    required = {
        "seed",
        "raw_action",
        "filtered_action",
        "chunk_starts",
    }
    traces = {}

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(
                    f"{path}: missing fields {sorted(missing)}"
                )

            seed = int(scalar(data["seed"]))
            raw = np.asarray(
                data["raw_action"],
                dtype=np.float64,
            )
            filtered = np.asarray(
                data["filtered_action"],
                dtype=np.float64,
            )
            chunk_starts = np.asarray(
                data["chunk_starts"],
                dtype=np.int64,
            ).reshape(-1)

            if seed in traces:
                raise RuntimeError(
                    f"Duplicate trace seed: {seed}"
                )
            if raw.ndim != 2:
                raise RuntimeError(
                    f"{path}: raw_action must be 2-D, "
                    f"got {raw.shape}"
                )
            if filtered.shape != raw.shape:
                raise RuntimeError(
                    f"{path}: raw/filtered shape mismatch: "
                    f"{raw.shape} vs {filtered.shape}"
                )
            if raw.shape[0] < 4:
                raise RuntimeError(
                    f"{path}: trajectory is too short "
                    f"for D3 analysis"
                )
            if not np.isfinite(raw).all():
                raise RuntimeError(
                    f"{path}: raw_action contains non-finite values"
                )
            if not np.isfinite(filtered).all():
                raise RuntimeError(
                    f"{path}: filtered_action contains "
                    "non-finite values"
                )
            if chunk_starts.ndim != 1:
                raise RuntimeError(
                    f"{path}: chunk_starts must be 1-D"
                )
            if chunk_starts.size == 0:
                raise RuntimeError(
                    f"{path}: chunk_starts is empty"
                )
            if int(chunk_starts[0]) != 0:
                raise RuntimeError(
                    f"{path}: first chunk start must be 0"
                )
            if np.any(np.diff(chunk_starts) <= 0):
                raise RuntimeError(
                    f"{path}: chunk_starts must be strictly "
                    "increasing"
                )
            if np.any(chunk_starts < 0) or np.any(
                chunk_starts >= raw.shape[0]
            ):
                raise RuntimeError(
                    f"{path}: chunk_starts outside trajectory"
                )

            score = None
            if "score" in data.files:
                score = float(scalar(data["score"]))

            traces[seed] = {
                "path": str(path),
                "seed": seed,
                "score": score,
                "raw": raw,
                "filtered": filtered,
                "chunk_starts": chunk_starts,
            }

    actual = tuple(sorted(traces))
    if actual != EXPECTED_ALL:
        missing = sorted(set(EXPECTED_ALL) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_ALL))
        raise RuntimeError(
            "Unexpected seed set. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    return traces


def boundary_mask(length, order, chunk_starts):
    """
    For a kth finite difference ending at action index t, mark it
    boundary-affected when its support [t-k, ..., t] crosses at least
    one nonzero chunk start.
    """
    endpoints = np.arange(order, length, dtype=np.int64)
    nonzero_starts = chunk_starts[chunk_starts > 0]

    if nonzero_starts.size == 0:
        return np.zeros(endpoints.size, dtype=bool)

    mask = np.zeros(endpoints.size, dtype=bool)
    for start in nonzero_starts:
        mask |= (
            (endpoints - order < start)
            & (start <= endpoints)
        )
    return mask


def difference_norms(actions, order):
    diff = np.diff(actions, n=order, axis=0)
    return np.linalg.norm(diff, axis=1)


def safe_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def safe_median(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def relative_reduction(raw_value, filtered_value):
    if (
        not np.isfinite(raw_value)
        or abs(raw_value) <= ZERO_TOL
    ):
        return float("nan")
    return float(
        100.0 * (raw_value - filtered_value) / raw_value
    )


def episode_metrics(trace):
    raw = trace["raw"]
    filtered = trace["filtered"]
    chunk_starts = trace["chunk_starts"]
    seed = trace["seed"]

    first_raw = raw[chunk_starts]
    first_filtered = filtered[chunk_starts]
    first_abs = np.abs(first_filtered - first_raw)
    first_max = (
        float(np.max(first_abs))
        if first_abs.size
        else 0.0
    )
    first_violations = int(
        np.count_nonzero(first_abs > FIRST_ACTION_ATOL)
    )
    exact_equal = bool(
        np.array_equal(first_raw, first_filtered)
    )

    row = {
        "seed": seed,
        "score": trace["score"],
        "n_actions": int(raw.shape[0]),
        "action_dim": int(raw.shape[1]),
        "n_chunks": int(chunk_starts.size),
        "first_actions_checked": int(
            chunk_starts.size * raw.shape[1]
        ),
        "first_action_exact_equal": exact_equal,
        "first_action_max_abs_diff": first_max,
        "first_action_violations": first_violations,
    }

    for order in (1, 2, 3):
        raw_norm = difference_norms(raw, order)
        filtered_norm = difference_norms(
            filtered,
            order,
        )
        affected = boundary_mask(
            raw.shape[0],
            order,
            chunk_starts,
        )
        within = ~affected

        if not np.any(within):
            raise RuntimeError(
                f"Seed {seed}: no within-chunk D{order} terms"
            )
        if not np.any(affected):
            raise RuntimeError(
                f"Seed {seed}: no boundary-affected "
                f"D{order} terms"
            )

        for region, mask in (
            ("within_chunk", within),
            ("boundary_affected", affected),
        ):
            raw_mean = float(np.mean(raw_norm[mask]))
            filtered_mean = float(
                np.mean(filtered_norm[mask])
            )
            absolute_reduction = raw_mean - filtered_mean
            percent_reduction = relative_reduction(
                raw_mean,
                filtered_mean,
            )

            prefix = f"d{order}_{region}"
            row[f"{prefix}_count"] = int(
                np.count_nonzero(mask)
            )
            row[f"{prefix}_raw"] = raw_mean
            row[f"{prefix}_filtered"] = filtered_mean
            row[
                f"{prefix}_absolute_reduction"
            ] = absolute_reduction
            row[
                f"{prefix}_percent_reduction"
            ] = percent_reduction

        row[
            f"d{order}_within_minus_boundary_pp"
        ] = (
            row[
                f"d{order}_within_chunk_percent_reduction"
            ]
            - row[
                f"d{order}_boundary_affected_percent_reduction"
            ]
        )

    return row


def bootstrap_mean_ci(
    values,
    samples=BOOTSTRAP_SAMPLES,
    seed=BASE_BOOTSTRAP_SEED,
):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value

    rng = np.random.default_rng(seed)
    batch = 10000
    means = np.empty(samples, dtype=np.float64)
    offset = 0

    while offset < samples:
        take = min(batch, samples - offset)
        indices = rng.integers(
            0,
            arr.size,
            size=(take, arr.size),
        )
        means[offset:offset + take] = np.mean(
            arr[indices],
            axis=1,
        )
        offset += take

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )
    return float(lower), float(upper)


def rank_biserial(diff):
    arr = np.asarray(diff, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) > ZERO_TOL]
    if arr.size == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(arr), method="average")
    positive = float(np.sum(ranks[arr > 0]))
    negative = float(np.sum(ranks[arr < 0]))
    total = positive + negative
    if total <= 0:
        return 0.0
    return (positive - negative) / total


def wilcoxon_p(diff):
    arr = np.asarray(diff, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[np.abs(arr) > ZERO_TOL]

    if nonzero.size == 0:
        return 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.wilcoxon(
            nonzero,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="approx",
        )
    return float(result.pvalue)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (m - rank) * float(p[index]),
        )
        running = max(running, candidate)
        adjusted[index] = running

    return adjusted


def extract(rows, key):
    return np.asarray(
        [row[key] for row in rows],
        dtype=np.float64,
    )


def paired_test(
    rows,
    order,
    comparison,
    bootstrap_seed,
):
    if comparison == "within_raw_vs_filtered":
        region = "within_chunk"
        raw = extract(rows, f"d{order}_{region}_raw")
        filtered = extract(
            rows,
            f"d{order}_{region}_filtered",
        )
        diff = raw - filtered
        relative = 100.0 * diff / raw
        label = "within_chunk: raw - filtered"
        positive_meaning = "AASF is smoother within chunks"
    elif comparison == "boundary_raw_vs_filtered":
        region = "boundary_affected"
        raw = extract(rows, f"d{order}_{region}_raw")
        filtered = extract(
            rows,
            f"d{order}_{region}_filtered",
        )
        diff = raw - filtered
        relative = 100.0 * diff / raw
        label = "boundary_affected: raw - filtered"
        positive_meaning = (
            "AASF is smoother on boundary-affected terms"
        )
    elif comparison == "within_vs_boundary":
        within = extract(
            rows,
            f"d{order}_within_chunk_percent_reduction",
        )
        boundary = extract(
            rows,
            (
                f"d{order}_boundary_affected_"
                "percent_reduction"
            ),
        )
        diff = within - boundary
        relative = diff.copy()
        raw = within
        filtered = boundary
        label = (
            "within percent reduction - boundary "
            "percent reduction"
        )
        positive_meaning = (
            "AASF smoothing is stronger within chunks"
        )
    else:
        raise ValueError(comparison)

    finite = (
        np.isfinite(raw)
        & np.isfinite(filtered)
        & np.isfinite(diff)
        & np.isfinite(relative)
    )
    raw = raw[finite]
    filtered = filtered[finite]
    diff = diff[finite]
    relative = relative[finite]

    ci_low, ci_high = bootstrap_mean_ci(
        diff,
        seed=bootstrap_seed,
    )
    rel_ci_low, rel_ci_high = bootstrap_mean_ci(
        relative,
        seed=bootstrap_seed + 10000,
    )

    return {
        "order": order,
        "metric": f"D{order}",
        "comparison": comparison,
        "label": label,
        "positive_meaning": positive_meaning,
        "n": int(diff.size),
        "first_mean": float(np.mean(raw)),
        "second_mean": float(np.mean(filtered)),
        "mean_difference": float(np.mean(diff)),
        "difference_ci_low": ci_low,
        "difference_ci_high": ci_high,
        "mean_relative_difference_percent": float(
            np.mean(relative)
        ),
        "relative_ci_low_percent": rel_ci_low,
        "relative_ci_high_percent": rel_ci_high,
        "positive_count": int(
            np.count_nonzero(diff > ZERO_TOL)
        ),
        "tie_count": int(
            np.count_nonzero(np.abs(diff) <= ZERO_TOL)
        ),
        "negative_count": int(
            np.count_nonzero(diff < -ZERO_TOL)
        ),
        "wilcoxon_p_raw": wilcoxon_p(diff),
        "rank_biserial": rank_biserial(diff),
    }


def subset_rows(all_rows, seeds):
    seed_set = set(seeds)
    rows = [
        row for row in all_rows
        if row["seed"] in seed_set
    ]
    if len(rows) != len(seed_set):
        raise RuntimeError(
            f"Expected {len(seed_set)} rows, got {len(rows)}"
        )
    return sorted(rows, key=lambda item: item["seed"])


def analyze_subset(name, rows, seed_offset):
    summaries = []
    tests = []

    first_checked = sum(
        row["first_actions_checked"] for row in rows
    )
    first_violations = sum(
        row["first_action_violations"] for row in rows
    )
    first_max = max(
        row["first_action_max_abs_diff"] for row in rows
    )
    exact_episodes = sum(
        bool(row["first_action_exact_equal"])
        for row in rows
    )

    for order in (1, 2, 3):
        for region in (
            "within_chunk",
            "boundary_affected",
        ):
            prefix = f"d{order}_{region}"
            raw = extract(rows, f"{prefix}_raw")
            filtered = extract(
                rows,
                f"{prefix}_filtered",
            )
            absolute = raw - filtered
            relative = 100.0 * absolute / raw

            summaries.append({
                "subset": name,
                "metric": f"D{order}",
                "region": region,
                "n_episodes": len(rows),
                "total_terms": int(sum(
                    row[f"{prefix}_count"]
                    for row in rows
                )),
                "raw_mean": safe_mean(raw),
                "raw_std": safe_std(raw),
                "raw_median": safe_median(raw),
                "filtered_mean": safe_mean(filtered),
                "filtered_std": safe_std(filtered),
                "filtered_median": safe_median(
                    filtered
                ),
                "mean_absolute_reduction": safe_mean(
                    absolute
                ),
                "mean_percent_reduction": safe_mean(
                    relative
                ),
            })

        comparisons = (
            "within_raw_vs_filtered",
            "boundary_raw_vs_filtered",
            "within_vs_boundary",
        )
        for index, comparison in enumerate(comparisons):
            result = paired_test(
                rows,
                order,
                comparison,
                (
                    BASE_BOOTSTRAP_SEED
                    + seed_offset
                    + order * 100
                    + index
                ),
            )
            result["subset"] = name
            tests.append(result)

    adjusted = holm_adjust([
        item["wilcoxon_p_raw"]
        for item in tests
    ])
    for item, value in zip(tests, adjusted):
        item["wilcoxon_p_holm_9"] = float(value)

    audit = {
        "subset": name,
        "n_episodes": len(rows),
        "first_actions_checked": int(first_checked),
        "first_action_violations": int(
            first_violations
        ),
        "episodes_exact_equal": int(exact_episodes),
        "max_abs_difference": float(first_max),
        "tolerance": FIRST_ACTION_ATOL,
        "passed": bool(
            first_violations == 0
            and exact_episodes == len(rows)
        ),
    }

    return {
        "audit": audit,
        "summaries": summaries,
        "tests": tests,
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: python_value(row.get(key))
                for key in fieldnames
            })


def fmt(value, digits=6):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def scientific(value):
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def build_report(primary, pooled):
    lines = []
    lines.append(
        "Chunk-internal versus chunk-boundary "
        "AASF analysis"
    )
    lines.append("=" * 64)
    lines.append("")
    lines.append(
        "Definition: Dk = mean_t ||Delta^k a_t||_2."
    )
    lines.append(
        "A kth difference is boundary-affected when "
        "its support crosses a nonzero chunk start."
    )
    lines.append(
        "Positive raw-minus-filtered differences mean "
        "that AASF is smoother."
    )
    lines.append(
        "Positive within-minus-boundary percentage-point "
        "differences mean stronger smoothing inside chunks."
    )
    lines.append("")

    for title, result in (
        ("Held-out 50 primary results", primary),
        ("Pooled 100 secondary results", pooled),
    ):
        lines.append(title)
        lines.append("-" * len(title))
        audit = result["audit"]
        lines.append(
            "First-action audit: "
            f"{audit['first_actions_checked']} scalar values, "
            f"violations={audit['first_action_violations']}, "
            f"max_abs_diff={audit['max_abs_difference']:.9g}, "
            f"passed={audit['passed']}"
        )
        lines.append("")

        summary_lookup = {
            (item["metric"], item["region"]): item
            for item in result["summaries"]
        }
        test_lookup = {
            (item["metric"], item["comparison"]): item
            for item in result["tests"]
        }

        for order in (1, 2, 3):
            metric = f"D{order}"
            lines.append(metric)

            for region, comparison in (
                (
                    "within_chunk",
                    "within_raw_vs_filtered",
                ),
                (
                    "boundary_affected",
                    "boundary_raw_vs_filtered",
                ),
            ):
                summary = summary_lookup[
                    (metric, region)
                ]
                test = test_lookup[
                    (metric, comparison)
                ]
                lines.append(
                    f"  {region}: raw="
                    f"{summary['raw_mean']:.8f}, filtered="
                    f"{summary['filtered_mean']:.8f}, "
                    f"mean reduction="
                    f"{test['mean_difference']:+.8f} "
                    f"({test['mean_relative_difference_percent']:+.3f}%), "
                    f"95% CI=[{test['difference_ci_low']:+.8f}, "
                    f"{test['difference_ci_high']:+.8f}], "
                    f"positive/tie/negative="
                    f"{test['positive_count']}/"
                    f"{test['tie_count']}/"
                    f"{test['negative_count']}, "
                    f"Wilcoxon p(raw)="
                    f"{test['wilcoxon_p_raw']:.9g}, "
                    f"p(Holm across 9)="
                    f"{test['wilcoxon_p_holm_9']:.9g}, "
                    f"rank-biserial="
                    f"{test['rank_biserial']:+.6f}"
                )

            mechanism = test_lookup[
                (metric, "within_vs_boundary")
            ]
            lines.append(
                "  within-vs-boundary smoothing strength: "
                f"within={mechanism['first_mean']:+.3f}%, "
                f"boundary={mechanism['second_mean']:+.3f}%, "
                f"advantage="
                f"{mechanism['mean_difference']:+.3f} pp, "
                f"95% CI=["
                f"{mechanism['difference_ci_low']:+.3f}, "
                f"{mechanism['difference_ci_high']:+.3f}] pp, "
                f"positive/tie/negative="
                f"{mechanism['positive_count']}/"
                f"{mechanism['tie_count']}/"
                f"{mechanism['negative_count']}, "
                f"Wilcoxon p(raw)="
                f"{mechanism['wilcoxon_p_raw']:.9g}, "
                f"p(Holm across 9)="
                f"{mechanism['wilcoxon_p_holm_9']:.9g}, "
                f"rank-biserial="
                f"{mechanism['rank_biserial']:+.6f}"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def build_latex(primary):
    summary_lookup = {
        (item["metric"], item["region"]): item
        for item in primary["summaries"]
    }
    test_lookup = {
        (item["metric"], item["comparison"]): item
        for item in primary["tests"]
    }

    lines = [
        r"% Auto-generated held-out chunk-region table.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Held-out chunk-internal and "
        r"chunk-boundary AASF smoothness analysis.}",
        r"\label{tab:chunk_boundary_analysis}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Metric & Region & Raw & Filtered & "
        r"Reduction (\%) & Holm $p$ \\",
        r"\hline",
    ]

    for order in (1, 2, 3):
        metric = f"D{order}"
        for region, comparison, display in (
            (
                "within_chunk",
                "within_raw_vs_filtered",
                "Within chunk",
            ),
            (
                "boundary_affected",
                "boundary_raw_vs_filtered",
                "Boundary affected",
            ),
        ):
            summary = summary_lookup[
                (metric, region)
            ]
            test = test_lookup[
                (metric, comparison)
            ]
            lines.append(
                f"{metric} & {display} & "
                f"{summary['raw_mean']:.4f} & "
                f"{summary['filtered_mean']:.4f} & "
                f"{test['mean_relative_difference_percent']:+.2f} & "
                f"{test['wilcoxon_p_holm_9']:.3g} \\\\"
            )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"% Mechanism comparison: positive values mean "
        r"stronger smoothing within chunks.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Held-out within-versus-boundary "
        r"smoothing-strength comparison.}",
        r"\label{tab:chunk_boundary_mechanism}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Metric & Advantage (pp) & 95\% CI (pp) & "
        r"Holm $p$ \\",
        r"\hline",
    ])

    for order in (1, 2, 3):
        metric = f"D{order}"
        test = test_lookup[
            (metric, "within_vs_boundary")
        ]
        lines.append(
            f"{metric} & "
            f"{test['mean_difference']:+.2f} & "
            f"[{test['difference_ci_low']:+.2f}, "
            f"{test['difference_ci_high']:+.2f}] & "
            f"{test['wilcoxon_p_holm_9']:.3g} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def freeze_outputs(
    project_root,
    output_dir,
    output_paths,
):
    manifest = output_dir / (
        "chunk_boundary_analysis_outputs.sha256"
    )
    lines = []
    for path in sorted(output_paths):
        relative = path.relative_to(project_root)
        lines.append(
            f"{sha256_file(path)}  {relative.as_posix()}"
        )
    manifest.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return manifest


def print_primary(primary):
    print("===== Held-out 50 primary results =====")
    audit = primary["audit"]
    print(
        "First-action audit: "
        f"episodes_exact_equal="
        f"{audit['episodes_exact_equal']}/"
        f"{audit['n_episodes']}, "
        f"scalar_violations="
        f"{audit['first_action_violations']}, "
        f"max_abs_diff="
        f"{audit['max_abs_difference']:.9g}, "
        f"PASS={audit['passed']}"
    )
    print()

    summary_lookup = {
        (item["metric"], item["region"]): item
        for item in primary["summaries"]
    }
    test_lookup = {
        (item["metric"], item["comparison"]): item
        for item in primary["tests"]
    }

    for order in (1, 2, 3):
        metric = f"D{order}"
        print(metric)
        for region, comparison in (
            (
                "within_chunk",
                "within_raw_vs_filtered",
            ),
            (
                "boundary_affected",
                "boundary_raw_vs_filtered",
            ),
        ):
            summary = summary_lookup[
                (metric, region)
            ]
            test = test_lookup[
                (metric, comparison)
            ]
            print(
                f"  {region}: "
                f"raw={summary['raw_mean']:.8f}, "
                f"filtered={summary['filtered_mean']:.8f}"
            )
            print(
                "    mean reduction="
                f"{test['mean_difference']:+.8f} "
                f"({test['mean_relative_difference_percent']:+.3f}%), "
                "bootstrap 95% CI=["
                f"{test['difference_ci_low']:+.8f}, "
                f"{test['difference_ci_high']:+.8f}]"
            )
            print(
                "    smoother/tie/rougher="
                f"{test['positive_count']}/"
                f"{test['tie_count']}/"
                f"{test['negative_count']}"
            )
            print(
                "    Wilcoxon p(raw)="
                f"{test['wilcoxon_p_raw']:.9g}, "
                "p(Holm across 9)="
                f"{test['wilcoxon_p_holm_9']:.9g}, "
                "rank-biserial="
                f"{test['rank_biserial']:+.6f}"
            )

        mechanism = test_lookup[
            (metric, "within_vs_boundary")
        ]
        print(
            "  within-vs-boundary smoothing strength:"
        )
        print(
            f"    within={mechanism['first_mean']:+.3f}%, "
            f"boundary={mechanism['second_mean']:+.3f}%, "
            f"advantage="
            f"{mechanism['mean_difference']:+.3f} pp, "
            "bootstrap 95% CI=["
            f"{mechanism['difference_ci_low']:+.3f}, "
            f"{mechanism['difference_ci_high']:+.3f}] pp"
        )
        print(
            "    stronger/tie/weaker="
            f"{mechanism['positive_count']}/"
            f"{mechanism['tie_count']}/"
            f"{mechanism['negative_count']}"
        )
        print(
            "    Wilcoxon p(raw)="
            f"{mechanism['wilcoxon_p_raw']:.9g}, "
            "p(Holm across 9)="
            f"{mechanism['wilcoxon_p_holm_9']:.9g}, "
            "rank-biserial="
            f"{mechanism['rank_biserial']:+.6f}"
        )
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Diffusion Policy project root. By default, "
            "the script uses the parent directory of this "
            "package."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else script_dir.parent.resolve()
    )
    trace_dir = project_root / TRACE_RELATIVE
    output_dir = project_root / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==== Chunk-boundary analysis preflight ====")
    print(f"Project root : {project_root}")
    print(f"Trace dir    : {trace_dir}")
    print(f"Output dir   : {output_dir}")
    print(
        f"Bootstrap    : {BOOTSTRAP_SAMPLES} paired resamples"
    )
    print()

    if not trace_dir.is_dir():
        raise SystemExit(
            "Missing AASF trace directory: "
            f"{trace_dir}"
        )

    traces = load_traces(trace_dir)
    all_rows = [
        episode_metrics(traces[seed])
        for seed in EXPECTED_ALL
    ]

    heldout_rows = subset_rows(
        all_rows,
        EXPECTED_HOLDOUT,
    )
    pooled_rows = subset_rows(
        all_rows,
        EXPECTED_ALL,
    )

    primary = analyze_subset(
        "heldout_4300050_4300099",
        heldout_rows,
        0,
    )
    pooled = analyze_subset(
        "pooled_4300000_4300099",
        pooled_rows,
        5000,
    )

    episode_csv = output_dir / (
        "chunk_boundary_episode_metrics.csv"
    )
    summary_csv = output_dir / (
        "chunk_boundary_summary.csv"
    )
    tests_csv = output_dir / (
        "chunk_boundary_paired_tests.csv"
    )
    audit_csv = output_dir / (
        "chunk_boundary_first_action_audit.csv"
    )
    statistics_json = output_dir / (
        "chunk_boundary_statistics.json"
    )
    report_txt = output_dir / (
        "chunk_boundary_report.txt"
    )
    latex_tex = output_dir / (
        "chunk_boundary_paper_tables.tex"
    )

    write_csv(episode_csv, all_rows)
    write_csv(
        summary_csv,
        primary["summaries"] + pooled["summaries"],
    )
    write_csv(
        tests_csv,
        primary["tests"] + pooled["tests"],
    )
    write_csv(
        audit_csv,
        [primary["audit"], pooled["audit"]],
    )

    payload = {
        "analysis": (
            "AASF chunk-internal versus "
            "chunk-boundary smoothness"
        ),
        "definitions": {
            "metric": (
                "Dk = mean_t ||Delta^k a_t||_2"
            ),
            "within_chunk": (
                "Finite-difference support does not cross "
                "a nonzero chunk start."
            ),
            "boundary_affected": (
                "Finite-difference support crosses at least "
                "one nonzero chunk start."
            ),
            "primary_subset": list(EXPECTED_HOLDOUT),
            "secondary_subset": list(EXPECTED_ALL),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "holm_family": (
                "Nine planned held-out tests: D1-D3 x "
                "{within raw-vs-filtered, boundary "
                "raw-vs-filtered, within-vs-boundary "
                "relative reduction}."
            ),
        },
        "primary": primary,
        "secondary": pooled,
    }
    statistics_json.write_text(
        json.dumps(
            json_ready(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_txt.write_text(
        build_report(primary, pooled),
        encoding="utf-8",
    )
    latex_tex.write_text(
        build_latex(primary),
        encoding="utf-8",
    )

    output_paths = [
        episode_csv,
        summary_csv,
        tests_csv,
        audit_csv,
        statistics_json,
        report_txt,
        latex_tex,
    ]
    manifest = freeze_outputs(
        project_root,
        output_dir,
        output_paths,
    )

    print_primary(primary)

    print("Saved files:")
    for path in output_paths:
        print(
            "  "
            + str(path.relative_to(project_root))
        )
    print(
        "  "
        + str(manifest.relative_to(project_root))
    )
    print()

    if not primary["audit"]["passed"]:
        raise SystemExit(
            "First-action audit failed on held-out traces."
        )
    if not pooled["audit"]["passed"]:
        raise SystemExit(
            "First-action audit failed on pooled traces."
        )

    print(
        "✅ Chunk-region analysis completed and frozen."
    )


if __name__ == "__main__":
    main()
