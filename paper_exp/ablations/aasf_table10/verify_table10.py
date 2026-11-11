#!/usr/bin/env python3
"""Recompute the source fields of manuscript Table 10 from 50-seed logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
SEEDS = range(4300000, 4300050)
TOLERANCE = 1e-12

LOGS = {
    "Baseline": "original.json",
    "ASF_alpha_0.7": "ema_alpha_0p7.json",
    "AASF_th8": "aasf_tau_8.json",
    "AASF_th10": "aasf_gamma_0p25.json",
    "AASF_th12": "aasf_tau_12.json",
    "AASF_gain0.15": "aasf_gamma_0p15.json",
    "AASF_gain0.25": "aasf_gamma_0p25.json",
    "AASF_gain0.35": "aasf_gamma_0p35.json",
}

METRICS = {
    "mean_score": "sim_max_reward",
    "action_delta_l2_mean": "action_delta_l2_mean",
    "action_acc_l2_mean": "action_acc_l2_mean",
    "action_jerk_l2_mean": "action_jerk_l2_mean",
}


def load_log(filename: str) -> dict[str, float]:
    path = ROOT / "raw_eval_logs" / filename
    return json.loads(path.read_text())


def seed_values(data: dict[str, float], metric: str) -> list[float]:
    values = []
    for seed in SEEDS:
        key = f"test/{metric}_{seed}"
        if key not in data:
            raise RuntimeError(f"Missing key {key}")
        values.append(float(data[key]))
    return values


def recompute(filename: str) -> dict[str, float]:
    data = load_log(filename)
    scores = seed_values(data, "sim_max_reward")
    result = {
        column: mean(seed_values(data, metric))
        for column, metric in METRICS.items()
    }
    result["success_rate_0.90"] = sum(x >= 0.90 for x in scores) / 50
    result["success_rate_0.95"] = sum(x >= 0.95 for x in scores) / 50
    result["failed_cases_below_0.90"] = sum(x < 0.90 for x in scores)
    return result


def verify_table(path: Path) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    checked = 0
    for row in rows:
        method = row["method"]
        if method not in LOGS:
            raise RuntimeError(f"No raw-log mapping for {method}")
        actual = recompute(LOGS[method])
        for column, value in actual.items():
            expected = float(row[column])
            if abs(expected - value) > TOLERANCE:
                raise RuntimeError(
                    f"{path.name}: {method} {column}: "
                    f"expected {expected}, recomputed {value}"
                )
        checked += 1

    print(f"PASS {path.name}: {checked} rows")


def main() -> None:
    verify_table(ROOT / "tables" / "aasf_threshold_ablation.csv")
    verify_table(ROOT / "tables" / "aasf_gain_ablation.csv")
    print("AASF_TABLE10_RAW_LOG_AUDIT = PASS")


if __name__ == "__main__":
    main()
