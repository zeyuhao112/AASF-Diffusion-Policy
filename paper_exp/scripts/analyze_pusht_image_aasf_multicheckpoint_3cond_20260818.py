from pathlib import Path
import csv
import json
import re
import hashlib
import subprocess
import numpy as np

OUT = Path(
    "paper_exp/tables/"
    "pusht_image_aasf_multicheckpoint_3cond_20260818"
)

BOOTSTRAP_SAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260818
DT = 0.1

CONDITIONS = [
    {
        "condition": "Push-T image seed42 / epoch0500",
        "short": "IMG-42",
        "run": "pusht_image_aasf_multicheckpoint_seed42_fresh50",
        "seeds": list(range(4700100, 4700150)),
        "checkpoint": "epoch0500",
        "new_or_existing": "new",
    },
    {
        "condition": "Push-T image seed43 / epoch1450",
        "short": "IMG-43",
        "run": "pusht_image_aasf_multicheckpoint_seed43_fresh50",
        "seeds": list(range(4700100, 4700150)),
        "checkpoint": "epoch1450",
        "new_or_existing": "new_resume",
    },
    {
        "condition": "Push-T image seed44 / epoch1850",
        "short": "IMG-44",
        "run": "pusht_image_aasf_confirmatory_fresh50",
        "seeds": list(range(4700000, 4700050)),
        "checkpoint": "epoch1850",
        "new_or_existing": "existing_frozen",
    },
]

FREEZES = [
    Path(
        "paper_artifacts/final/"
        "image_aasf_multicheckpoint_replication_freeze_20260818/"
        "new_seed42_seed43_artifacts.sha256"
    ),
    Path(
        "paper_artifacts/final/"
        "image_aasf_confirmatory_freeze_20260814/"
        "all_confirmatory_files.sha256"
    ),
    Path(
        "paper_artifacts/final/"
        "image_aasf_confirmatory_freeze_20260814/"
        "analysis_outputs.sha256"
    ),
    Path(
        "paper_artifacts/final/"
        "image_aasf_confirmatory_freeze_20260814/"
        "core_identity.sha256"
    ),
]


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
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
            f"manifest verification failed: {path}"
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
            "bootstrap input has non-finite values"
        )

    n = values.size
    rng = np.random.default_rng(int(seed))

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

        means[done:done + m] = np.mean(
            values[idx],
            axis=1,
        )

        done += m

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


def finite_difference_metric(x, order, dt=1.0):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"trajectory must be 2D, got {x.shape}"
        )

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


def pct_reduction(original, aasf, name):
    original = float(original)
    aasf = float(aasf)

    if (
        not np.isfinite(original)
        or not np.isfinite(aasf)
        or original <= 0
    ):
        raise RuntimeError(
            f"{name}: invalid percent reduction base"
        )

    return float(
        100.0 * (original - aasf) / original
    )


def load_eval_scores(run, method):
    result_root = Path("paper_exp/results") / run

    candidates = []

    direct = result_root / method / "eval_log.json"

    if direct.is_file():
        candidates.append(direct)

    for path in sorted(
        result_root.glob(f"{method}*/eval_log.json")
    ):
        if path not in candidates:
            candidates.append(path)

    by_seed = {}

    for path in candidates:
        data = json.loads(path.read_text())

        if not isinstance(data, dict):
            continue

        for key, value in data.items():
            m = re.match(
                r"^test/sim_max_reward_(\d+)$",
                str(key),
            )

            if not m:
                continue

            seed = int(m.group(1))
            score = float(value)

            if seed in by_seed:
                if abs(by_seed[seed] - score) > 1e-12:
                    raise RuntimeError(
                        f"conflicting score for seed {seed}: "
                        f"{by_seed[seed]} vs {score}"
                    )

            by_seed[seed] = score

    return by_seed




def load_trace(run, method, seed):
    path = (
        Path("paper_artifacts/traces")
        / run
        / method
        / f"seed_{seed}_{method}.npz"
    )

    if not path.is_file():
        raise RuntimeError(
            f"missing trace: {path}"
        )

    with np.load(path, allow_pickle=True) as d:
        actual = np.asarray(
            d["actual_executed_action"],
            dtype=np.float64,
        )

        if actual.ndim != 2:
            raise RuntimeError(
                f"{path}: actual action not 2D"
            )

        score = np.nan

        for key in (
            "score",
            "max_reward",
        ):
            if key in d.files:
                try:
                    score = float(
                        np.asarray(d[key]).reshape(())
                    )
                    break
                except Exception:
                    pass

        if not np.isfinite(score) and "reward" in d.files:
            reward = np.asarray(
                d["reward"],
                dtype=np.float64,
            )

            if reward.size > 0 and np.any(np.isfinite(reward)):
                score = float(np.nanmax(reward))

        actual_steps = actual.shape[0]

        chunks = np.asarray(d["executed_action"])
        steps = np.asarray(
            d["executed_steps_per_query"],
            dtype=np.int64,
        )

        rebuilt = np.concatenate(
            [
                chunks[i, :n]
                for i, n in enumerate(steps)
                if n > 0
            ],
            axis=0,
        )

        if (
            rebuilt.shape != actual.shape
            or not np.array_equal(rebuilt, actual)
        ):
            raise RuntimeError(
                f"{path}: execution accounting failed"
            )

        alpha = None

        for key in (
            "action_smooth_alpha",
            "alpha",
            "adaptive_alpha",
            "smooth_alpha",
        ):
            if key in d.files:
                arr = np.asarray(d[key])
                if arr.size > 0:
                    alpha = arr.astype(np.float64)
                    break

    return {
        "path": str(path),
        "actual": actual,
        "score": score,
        "actual_steps": actual_steps,
        "alpha": alpha,
    }




def analyze_condition(cond, condition_index):
    run = cond["run"]
    seeds = cond["seeds"]

    eval_scores = {
        method: load_eval_scores(run, method)
        for method in ("original", "aasf")
    }

    rows = []

    for seed in seeds:
        old = load_trace(
            run,
            "original",
            seed,
        )
        new = load_trace(
            run,
            "aasf",
            seed,
        )

        old_score = old["score"]
        new_score = new["score"]

        if seed in eval_scores["original"]:
            eval_score = eval_scores["original"][seed]

            if (
                np.isfinite(old_score)
                and abs(old_score - eval_score) > 1e-9
            ):
                raise RuntimeError(
                    f"{run} seed {seed}: original "
                    f"trace/eval score mismatch "
                    f"{old_score} vs {eval_score}"
                )

            old_score = eval_score

        if seed in eval_scores["aasf"]:
            eval_score = eval_scores["aasf"][seed]

            if (
                np.isfinite(new_score)
                and abs(new_score - eval_score) > 1e-9
            ):
                raise RuntimeError(
                    f"{run} seed {seed}: aasf "
                    f"trace/eval score mismatch "
                    f"{new_score} vs {eval_score}"
                )

            new_score = eval_score

        if not (
            np.isfinite(old_score)
            and np.isfinite(new_score)
        ):
            raise RuntimeError(
                f"{run} seed {seed}: missing score"
            )

        n = min(
            old["actual"].shape[0],
            new["actual"].shape[0],
        )

        if n < 4:
            raise RuntimeError(
                f"{run} seed {seed}: common prefix < 4"
            )

        o = old["actual"][:n]
        a = new["actual"][:n]

        row = {
            "condition": cond["condition"],
            "short": cond["short"],
            "run": run,
            "seed": seed,
            "original_score": old_score,
            "aasf_score": new_score,
            "score_diff": new_score - old_score,
            "original_sr90": float(old_score >= 0.90),
            "aasf_sr90": float(new_score >= 0.90),
            "sr90_diff": float(new_score >= 0.90)
            - float(old_score >= 0.90),
            "original_sr95": float(old_score >= 0.95),
            "aasf_sr95": float(new_score >= 0.95),
            "sr95_diff": float(new_score >= 0.95)
            - float(old_score >= 0.95),
            "original_exact1": float(old_score >= 1.0),
            "aasf_exact1": float(new_score >= 1.0),
            "exact1_diff": float(new_score >= 1.0)
            - float(old_score >= 1.0),
            "original_steps": old["actual_steps"],
            "aasf_steps": new["actual_steps"],
            "common_prefix_steps": n,
        }

        for order in (1, 2, 3):
            od = finite_difference_metric(
                o,
                order,
                dt=1.0,
            )
            ad = finite_difference_metric(
                a,
                order,
                dt=1.0,
            )

            oc = finite_difference_metric(
                o,
                order,
                dt=DT,
            )
            ac = finite_difference_metric(
                a,
                order,
                dt=DT,
            )

            dred = pct_reduction(
                od,
                ad,
                f"D{order}",
            )

            cred = pct_reduction(
                oc,
                ac,
                f"C{order}",
            )

            if not np.isclose(
                dred,
                cred,
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"{run} seed {seed}: "
                    f"D{order}/C{order} invariance failed"
                )

            row[f"original_D{order}"] = od
            row[f"aasf_D{order}"] = ad
            row[f"D{order}_reduction_pct"] = dred

            row[f"original_C{order}"] = oc
            row[f"aasf_C{order}"] = ac
            row[f"C{order}_reduction_pct"] = cred

        rows.append(row)

    stats = []

    def add_stat(metric, values):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        lo, hi = bootstrap_ci(
            values,
            BOOTSTRAP_BASE_SEED
            + 1000 * condition_index
            + len(stats)
            + 1,
        )

        stats.append(
            {
                "condition": cond["condition"],
                "short": cond["short"],
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    add_stat(
        "score_diff",
        [r["score_diff"] for r in rows],
    )

    add_stat(
        "sr90_risk_diff",
        [r["sr90_diff"] for r in rows],
    )

    add_stat(
        "sr95_risk_diff",
        [r["sr95_diff"] for r in rows],
    )

    add_stat(
        "exact1_risk_diff",
        [r["exact1_diff"] for r in rows],
    )

    for order in (1, 2, 3):
        vals = [
            r[f"D{order}_reduction_pct"]
            for r in rows
        ]

        add_stat(
            f"D{order}_reduction_pct",
            vals,
        )

        add_stat(
            f"C{order}_reduction_pct",
            vals,
        )

    return rows, stats


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("empty csv rows")

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        if any(OUT.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty analysis dir: {OUT}"
            )
    else:
        OUT.mkdir(
            parents=False,
            exist_ok=False,
        )

    for manifest in FREEZES:
        verify_manifest(manifest)

    all_rows = []
    all_stats = []

    for i, cond in enumerate(CONDITIONS):
        rows, stats = analyze_condition(
            cond,
            i,
        )

        all_rows.extend(rows)
        all_stats.extend(stats)

    write_csv(
        OUT / "aasf_image_3cond_per_seed.csv",
        all_rows,
    )

    write_csv(
        OUT / "aasf_image_3cond_stats.csv",
        all_stats,
    )

    by = {
        (r["short"], r["metric"]): r
        for r in all_stats
    }

    with open(
        OUT / "aasf_image_3cond_table.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "AASF zero-retuning image replication "
            "across three Push-T image checkpoints\n"
        )
        f.write(
            "=====================================================\n\n"
        )

        f.write(
            "Condition | ΔScore [95% CI] | SR@0.90 | "
            "D1/C1 | D2/C2 | D3/C3\n"
        )
        f.write(
            "-----------------------------------------------------\n"
        )

        for cond in CONDITIONS:
            s = cond["short"]

            score = by[(s, "score_diff")]
            sr90 = by[(s, "sr90_risk_diff")]
            d1 = by[(s, "D1_reduction_pct")]
            d2 = by[(s, "D2_reduction_pct")]
            d3 = by[(s, "D3_reduction_pct")]

            f.write(
                f"{cond['condition']} | "
                f"{score['mean']:+.4f} "
                f"[{score['ci95_low']:+.4f},"
                f"{score['ci95_high']:+.4f}] | "
                f"{100.0 * sr90['mean']:+.0f} pp "
                f"[{100.0 * sr90['ci95_low']:+.0f},"
                f"{100.0 * sr90['ci95_high']:+.0f}] | "
                f"{d1['mean']:+.2f}% "
                f"[{d1['ci95_low']:+.2f},"
                f"{d1['ci95_high']:+.2f}] | "
                f"{d2['mean']:+.2f}% "
                f"[{d2['ci95_low']:+.2f},"
                f"{d2['ci95_high']:+.2f}] | "
                f"{d3['mean']:+.2f}% "
                f"[{d3['ci95_low']:+.2f},"
                f"{d3['ci95_high']:+.2f}]\n"
            )

        f.write("\n")
        f.write(
            "Notes: positive D/C reductions indicate smoother "
            "AASF commands. C1-C3 percentage reductions are "
            "identical to D1-D3 because dt=0.1 is a positive "
            "constant scaling. BSOP is disabled in all rows.\n"
        )

    with open(
        OUT / "aasf_image_3cond_audit.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "===== AASF IMAGE 3-CONDITION ANALYSIS AUDIT =====\n\n"
        )
        f.write(
            "analysis_scope=offline_trace_analysis_only\n"
        )
        f.write("eval_py_called=NO\n")
        f.write("new_rollout=NO\n")
        f.write("conditions=IMG42,IMG43,IMG44\n")
        f.write("methods=original,aasf\n")
        f.write("BSOP_enabled=NO\n")
        f.write("image_specific_retuning=NO\n")
        f.write("checkpoint_specific_retuning=NO\n")
        f.write("AASF_alpha_min=0.45\n")
        f.write("AASF_alpha_max=0.80\n")
        f.write("AASF_tau=10.0\n")
        f.write("AASF_gamma=0.25\n")
        f.write("control_dt_s=0.1\n")
        f.write("D_C_reduction_invariance=PASS\n")
        f.write("freeze_manifest_verification=PASS\n")
        f.write(
            "seed43_resume_append_only=YES\n"
        )
        f.write(
            "seed42_new_traces=100\n"
        )
        f.write(
            "seed43_new_traces=100\n"
        )
        f.write(
            "seed44_existing_traces=100\n"
        )
        f.write(
            "\nAASF_IMAGE_3COND_ANALYSIS_AUDIT = PASS\n"
        )

    manifest = OUT / "analysis_artifact_sha256.txt"

    with open(
        manifest,
        "w",
        encoding="utf-8",
    ) as f:
        for p in [
            Path(__file__),
            OUT / "aasf_image_3cond_per_seed.csv",
            OUT / "aasf_image_3cond_stats.csv",
            OUT / "aasf_image_3cond_table.txt",
            OUT / "aasf_image_3cond_audit.txt",
            *FREEZES,
        ]:
            f.write(f"{sha(p)}  {p}\n")

    verify_manifest(manifest)

    print(
        "AASF_IMAGE_3COND_ANALYSIS_AUDIT = PASS"
    )
    print(
        "AASF_IMAGE_3COND_ANALYSIS_FINAL = PASS"
    )
    print()
    print(
        (OUT / "aasf_image_3cond_table.txt")
        .read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
