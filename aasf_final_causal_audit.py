#!/usr/bin/env python3
"""
Final causal audit for paired Original/AASF BlockPush forensic traces.

Expected diagnostic layout:
  <diag>/original/aasf_trace/seed_<SEED>_off.npz
  <diag>/aasf/aasf_trace/seed_<SEED>_adaptive.npz

Outputs:
  <diag>/AASF_FINAL_CAUSAL_CALLS.csv
  Human-readable report on stdout.

This script is read-only with respect to trace files. It does not rerun evaluation
or modify the policy/runner.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CASES = {
    4300024: "AASF loss",
    4300035: "AASF win",
    4300038: "AASF loss",
}

# Flattened low-dimensional observation layout used by the frozen BlockPush run.
# The ordering follows gym.spaces.Dict flattening in this environment.
OBS_SLICES = {
    "block2_translation": slice(1, 3),
    "block_translation": slice(4, 6),
    "effector_translation": slice(8, 10),
    "target2_translation": slice(11, 13),
    "target_translation": slice(14, 16),
}

TARGET_EVENT_KEYS = (
    "TARGET_0_0",
    "TARGET_0_1",
    "TARGET_1_0",
    "TARGET_1_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired Original/AASF BlockPush forensic traces."
    )
    parser.add_argument(
        "--diag",
        type=Path,
        default=None,
        help=(
            "Diagnostic directory. Defaults to $LATEST_DIAG; otherwise the "
            "newest paper_exp/diagnostics/blockpush_fullbatch_video_* directory."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="4300024,4300035,4300038",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Target-region center-distance threshold in metres (default: 0.05).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Environment control interval in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path. Defaults to <diag>/AASF_FINAL_CAUSAL_CALLS.csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of kinematic separation candidates to print.",
    )
    return parser.parse_args()


def locate_diag(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
    else:
        env_value = os.environ.get("LATEST_DIAG", "").strip()
        if env_value:
            root = Path(env_value).expanduser().resolve()
        else:
            base = Path("paper_exp/diagnostics")
            candidates = sorted(
                base.glob("blockpush_fullbatch_video_*"),
                key=lambda p: p.stat().st_mtime if p.exists() else -1.0,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    "No blockpush_fullbatch_video_* diagnostic directory found. "
                    "Set LATEST_DIAG or pass --diag."
                )
            root = candidates[0].resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def parse_seeds(text: str) -> List[int]:
    result: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError("No seeds were provided.")
    return result


def trace_path(root: Path, seed: int, branch: str) -> Path:
    if branch == "original":
        return root / "original" / "aasf_trace" / f"seed_{seed}_off.npz"
    if branch == "aasf":
        return root / "aasf" / "aasf_trace" / f"seed_{seed}_adaptive.npz"
    raise ValueError(f"Unknown branch: {branch}")


def load_trace(root: Path, seed: int, branch: str) -> Dict[str, np.ndarray]:
    path = trace_path(root, seed, branch)
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as data:
        trace = {key: data[key].copy() for key in data.files}

    required = {
        "policy_call_index",
        "environment_step_start",
        "obs_before",
        "raw_action",
        "executed_action",
        "obs_after",
        "reward",
        "done",
        "info",
    }
    missing = sorted(required.difference(trace))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    raw = np.asarray(trace["raw_action"])
    executed = np.asarray(trace["executed_action"])
    before = np.asarray(trace["obs_before"])
    after = np.asarray(trace["obs_after"])

    if raw.ndim != 3 or raw.shape[-1] != 2:
        raise RuntimeError(f"Unexpected raw_action shape in {path}: {raw.shape}")
    if executed.shape != raw.shape:
        raise RuntimeError(
            f"raw/executed shape mismatch in {path}: {raw.shape} vs {executed.shape}"
        )
    if before.ndim != 3 or before.shape[-1] != 16:
        raise RuntimeError(f"Unexpected obs_before shape in {path}: {before.shape}")
    if after.ndim != 3 or after.shape[-1] != 16:
        raise RuntimeError(f"Unexpected obs_after shape in {path}: {after.shape}")
    if not (raw.shape[0] == before.shape[0] == after.shape[0]):
        raise RuntimeError(f"Policy-call count mismatch in {path}")

    trace["_path"] = np.asarray(str(path))
    return trace


def decode(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    if flat.size != 16:
        raise RuntimeError(f"Expected 16 observation values, got {flat.size}")

    blocks = np.stack(
        [
            flat[OBS_SLICES["block_translation"]],
            flat[OBS_SLICES["block2_translation"]],
        ],
        axis=0,
    )
    targets = np.stack(
        [
            flat[OBS_SLICES["target_translation"]],
            flat[OBS_SLICES["target2_translation"]],
        ],
        axis=0,
    )
    effector = flat[OBS_SLICES["effector_translation"]]
    return blocks, targets, effector


def last_state(trace: Mapping[str, np.ndarray], key: str, call: int):
    obs = np.asarray(trace[key], dtype=np.float64)
    return decode(obs[call, -1])


def distance_matrix(blocks: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.linalg.norm(blocks[:, None, :] - targets[None, :, :], axis=-1)


def distinct_target_success(
    distances: np.ndarray, threshold: float
) -> Tuple[bool, np.ndarray, np.ndarray]:
    nearest = np.argmin(distances, axis=1).astype(np.int64)
    nearest_distance = distances[np.arange(2), nearest]
    success = bool(
        np.all(nearest_distance < threshold) and nearest[0] != nearest[1]
    )
    return success, nearest, nearest_distance


def bool_text(value: bool) -> str:
    return "YES" if value else "NO"


def finite(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    return "nan"


def flatten_numeric(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except Exception:
        return np.asarray([], dtype=np.float64)

    if arr.dtype == object:
        numbers: List[float] = []
        for item in arr.reshape(-1):
            try:
                nested = flatten_numeric(item)
                numbers.extend(nested.tolist())
            except Exception:
                continue
        return np.asarray(numbers, dtype=np.float64)

    try:
        return arr.astype(np.float64, copy=False).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)


def normalize_info_item(item: Any) -> Mapping[str, Any]:
    if isinstance(item, dict):
        return item

    if isinstance(item, np.ndarray) and item.shape == ():
        try:
            scalar = item.item()
        except Exception:
            return {}
        if isinstance(scalar, dict):
            return scalar

    try:
        converted = item.item()
    except Exception:
        return {}
    return converted if isinstance(converted, dict) else {}


def extract_event_steps(trace: Mapping[str, np.ndarray]) -> Dict[str, Optional[int]]:
    result: Dict[str, Optional[int]] = {key: None for key in TARGET_EVENT_KEYS}
    info_array = np.asarray(trace["info"], dtype=object).reshape(-1)

    for item in info_array:
        info = normalize_info_item(item)
        for key in TARGET_EVENT_KEYS:
            if key not in info:
                continue
            values = flatten_numeric(info[key])
            valid = values[np.isfinite(values) & (values >= 0)]
            if valid.size:
                step = int(np.min(valid))
                current = result[key]
                if current is None or step < current:
                    result[key] = step
    return result


def boundary_states(
    trace: Mapping[str, np.ndarray],
) -> List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    calls = int(np.asarray(trace["raw_action"]).shape[0])
    step_start = np.asarray(trace["environment_step_start"], dtype=np.int64)
    horizon = int(np.asarray(trace["raw_action"]).shape[1])

    result: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []

    blocks, targets, effector = last_state(trace, "obs_before", 0)
    result.append((0, int(step_start[0]), blocks, targets, effector))

    for call in range(calls):
        blocks, targets, effector = last_state(trace, "obs_after", call)
        if call + 1 < calls:
            env_step = int(step_start[call + 1])
        else:
            env_step = int(step_start[call]) + horizon
        result.append((call + 1, env_step, blocks, targets, effector))

    return result


def longest_true_streak(values: Sequence[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def boundary_occupancy_summary(
    trace: Mapping[str, np.ndarray], threshold: float
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    states = boundary_states(trace)
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for block_index in range(2):
        for target_index in range(2):
            flags: List[bool] = []
            distances: List[float] = []
            for _, _, blocks, targets, _ in states:
                distance = float(
                    np.linalg.norm(blocks[block_index] - targets[target_index])
                )
                distances.append(distance)
                flags.append(distance < threshold)

            entered = [i for i, flag in enumerate(flags) if flag]
            first_index = entered[0] if entered else None
            result[(block_index, target_index)] = {
                "first_boundary": (
                    states[first_index][0] if first_index is not None else None
                ),
                "first_env_step": (
                    states[first_index][1] if first_index is not None else None
                ),
                "occupied_boundaries": int(sum(flags)),
                "longest_streak": longest_true_streak(flags),
                "final_inside": bool(flags[-1]),
                "minimum_distance": float(np.min(distances)),
                "final_distance": float(distances[-1]),
                "flags": flags,
            }
    return result


def state_kinematics(
    trace: Mapping[str, np.ndarray], call: int, dt: float
) -> Dict[str, Any]:
    obs_after = np.asarray(trace["obs_after"], dtype=np.float64)
    sequence = obs_after[call]

    if sequence.shape[0] < 2:
        raise RuntimeError("At least two retained observations are required.")

    blocks0, targets0, effector0 = decode(sequence[-2])
    blocks1, targets1, effector1 = decode(sequence[-1])

    block_velocity = (blocks1 - blocks0) / dt
    effector_velocity = (effector1 - effector0) / dt
    block_speed = np.linalg.norm(block_velocity, axis=1)
    effector_speed = float(np.linalg.norm(effector_velocity))

    gap0 = np.linalg.norm(blocks0 - effector0[None, :], axis=1)
    gap1 = np.linalg.norm(blocks1 - effector1[None, :], axis=1)
    gap_rate = (gap1 - gap0) / dt

    d0 = distance_matrix(blocks0, targets0)
    d1 = distance_matrix(blocks1, targets1)
    nearest = np.argmin(d1, axis=1)
    radial_v = np.asarray(
        [(d1[b, nearest[b]] - d0[b, nearest[b]]) / dt for b in range(2)],
        dtype=np.float64,
    )

    return {
        "blocks0": blocks0,
        "blocks1": blocks1,
        "targets1": targets1,
        "effector0": effector0,
        "effector1": effector1,
        "block_speed": block_speed,
        "effector_speed": effector_speed,
        "gap": gap1,
        "gap_rate": gap_rate,
        "nearest_target": nearest,
        "radial_v": radial_v,
        "distances": d1,
    }


def scalar_at(trace: Mapping[str, np.ndarray], key: str, call: int) -> Any:
    arr = np.asarray(trace[key])
    value = arr[call]
    if np.asarray(value).shape == ():
        try:
            return np.asarray(value).item()
        except Exception:
            return value
    return value


def make_call_rows(
    seed: int,
    outcome: str,
    branch: str,
    trace: Mapping[str, np.ndarray],
    other_trace: Mapping[str, np.ndarray],
    threshold: float,
    dt: float,
    events: Mapping[str, Optional[int]],
) -> List[Dict[str, Any]]:
    raw = np.asarray(trace["raw_action"], dtype=np.float64)
    executed = np.asarray(trace["executed_action"], dtype=np.float64)
    other_raw = np.asarray(other_trace["raw_action"], dtype=np.float64)
    step_start = np.asarray(trace["environment_step_start"], dtype=np.int64)
    horizon = raw.shape[1]

    rows: List[Dict[str, Any]] = []

    for call in range(raw.shape[0]):
        before_blocks, before_targets, before_effector = last_state(
            trace, "obs_before", call
        )
        after_blocks, after_targets, after_effector = last_state(
            trace, "obs_after", call
        )

        before_dist = distance_matrix(before_blocks, before_targets)
        after_dist = distance_matrix(after_blocks, after_targets)
        success_before, nearest_before, nearest_dist_before = (
            distinct_target_success(before_dist, threshold)
        )
        success_after, nearest_after, nearest_dist_after = (
            distinct_target_success(after_dist, threshold)
        )

        kin = state_kinematics(trace, call, dt)

        filter_delta = executed[call] - raw[call]
        raw_pair_delta = raw[call] - other_raw[call]

        row: Dict[str, Any] = {
            "seed": seed,
            "outcome": outcome,
            "branch": branch,
            "call": call,
            "env_start": int(step_start[call]),
            "env_end_nominal": int(step_start[call]) + horizon - 1,
            "reward": scalar_at(trace, "reward", call),
            "done": scalar_at(trace, "done", call),
            "filter_l2_sum": float(
                np.sum(np.linalg.norm(filter_delta, axis=-1))
            ),
            "filter_l2_max": float(
                np.max(np.linalg.norm(filter_delta, axis=-1))
            ),
            "raw_pair_l2_sum": float(
                np.sum(np.linalg.norm(raw_pair_delta, axis=-1))
            ),
            "raw_pair_l2_max": float(
                np.max(np.linalg.norm(raw_pair_delta, axis=-1))
            ),
            "effector_x_before": float(before_effector[0]),
            "effector_y_before": float(before_effector[1]),
            "effector_x_after": float(after_effector[0]),
            "effector_y_after": float(after_effector[1]),
            "effector_speed_end": kin["effector_speed"],
            "success_before": success_before,
            "success_after": success_after,
        }

        for block_index in range(2):
            row[f"b{block_index}_x_before"] = float(before_blocks[block_index, 0])
            row[f"b{block_index}_y_before"] = float(before_blocks[block_index, 1])
            row[f"b{block_index}_x_after"] = float(after_blocks[block_index, 0])
            row[f"b{block_index}_y_after"] = float(after_blocks[block_index, 1])
            row[f"b{block_index}_nearest_before"] = int(nearest_before[block_index])
            row[f"b{block_index}_nearest_after"] = int(nearest_after[block_index])
            row[f"b{block_index}_nearest_dist_before"] = float(
                nearest_dist_before[block_index]
            )
            row[f"b{block_index}_nearest_dist_after"] = float(
                nearest_dist_after[block_index]
            )
            row[f"b{block_index}_effector_gap_before"] = float(
                np.linalg.norm(before_blocks[block_index] - before_effector)
            )
            row[f"b{block_index}_effector_gap_after"] = float(
                np.linalg.norm(after_blocks[block_index] - after_effector)
            )
            row[f"b{block_index}_speed_end"] = float(
                kin["block_speed"][block_index]
            )
            row[f"b{block_index}_gap_rate_end"] = float(
                kin["gap_rate"][block_index]
            )
            row[f"b{block_index}_radial_v_end"] = float(
                kin["radial_v"][block_index]
            )

            for target_index in range(2):
                row[f"b{block_index}_t{target_index}_dist_before"] = float(
                    before_dist[block_index, target_index]
                )
                row[f"b{block_index}_t{target_index}_dist_after"] = float(
                    after_dist[block_index, target_index]
                )
                row[f"event_TARGET_{block_index}_{target_index}"] = events[
                    f"TARGET_{block_index}_{target_index}"
                ]

        rows.append(row)

    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No CSV rows were generated.")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_call0(
    seed: int,
    outcome: str,
    original: Mapping[str, np.ndarray],
    aasf: Mapping[str, np.ndarray],
    tolerance: float = 1e-9,
) -> None:
    original_raw = np.asarray(original["raw_action"], dtype=np.float64)
    aasf_raw = np.asarray(aasf["raw_action"], dtype=np.float64)
    aasf_exec = np.asarray(aasf["executed_action"], dtype=np.float64)
    step_start = np.asarray(aasf["environment_step_start"], dtype=np.int64)

    raw_error = float(np.max(np.abs(aasf_raw[0] - original_raw[0])))
    delta = aasf_exec[0] - aasf_raw[0]
    delta_norm = np.linalg.norm(delta, axis=-1)
    changed = np.flatnonzero(delta_norm > tolerance)
    first_h = int(changed[0]) if changed.size else None

    o_before_b, o_before_t, o_before_e = last_state(original, "obs_before", 0)
    a_before_b, a_before_t, a_before_e = last_state(aasf, "obs_before", 0)
    o_after_b, o_after_t, o_after_e = last_state(original, "obs_after", 0)
    a_after_b, a_after_t, a_after_e = last_state(aasf, "obs_after", 0)

    print(f"\nSeed {seed} ({outcome})")
    print(f"  Initial raw-action max abs difference : {raw_error:.12f}")
    if first_h is None:
        print("  First executed-action modification    : none")
    else:
        print(
            "  First executed-action modification    : "
            f"h={first_h}, env step={int(step_start[0]) + first_h}, "
            f"|delta|={delta_norm[first_h]:.9f}"
        )
    print(
        "  Call-0 filter magnitude               : "
        f"sum={float(np.sum(delta_norm)):.9f}, "
        f"max={float(np.max(delta_norm)):.9f}"
    )
    print(
        "  Initial state mismatch (sanity check) : "
        f"blocks={float(np.max(np.abs(a_before_b-o_before_b))):.12f}, "
        f"targets={float(np.max(np.abs(a_before_t-o_before_t))):.12f}, "
        f"effector={float(np.max(np.abs(a_before_e-o_before_e))):.12f}"
    )
    print(
        "  End-of-call-0 branch state divergence : "
        f"b0={np.linalg.norm(a_after_b[0]-o_after_b[0]):.9f}, "
        f"b1={np.linalg.norm(a_after_b[1]-o_after_b[1]):.9f}, "
        f"effector={np.linalg.norm(a_after_e-o_after_e):.9f}"
    )

    print("  Modified call-0 actions:")
    print(
        f"    {'h':>2} {'env':>5} {'raw_x':>11} {'raw_y':>11} "
        f"{'exec_x':>11} {'exec_y':>11} {'d_x':>11} {'d_y':>11} {'|d|':>11}"
    )
    for h in range(delta.shape[0]):
        if delta_norm[h] <= tolerance and h != 0:
            continue
        env_step = int(step_start[0]) + h
        print(
            f"    {h:>2d} {env_step:>5d} "
            f"{aasf_raw[0,h,0]:>11.6f} {aasf_raw[0,h,1]:>11.6f} "
            f"{aasf_exec[0,h,0]:>11.6f} {aasf_exec[0,h,1]:>11.6f} "
            f"{delta[h,0]:>11.6f} {delta[h,1]:>11.6f} "
            f"{delta_norm[h]:>11.6f}"
        )


def print_dual_target_summary(
    seed: int,
    outcome: str,
    branch: str,
    trace: Mapping[str, np.ndarray],
    threshold: float,
) -> None:
    states = boundary_states(trace)
    print(f"\nSeed {seed} ({outcome}) | {branch}")
    print(
        f"  {'Boundary':>8} {'Env':>6} "
        f"{'b0-t0':>10} {'b0-t1':>10} {'b1-t0':>10} {'b1-t1':>10} "
        f"{'mapping':>12} {'success':>9}"
    )

    # Print B0, B1, B2, B3, every fifth later boundary, and final boundary.
    selected = set(range(min(4, len(states))))
    selected.update(range(5, len(states), 5))
    selected.add(len(states) - 1)

    for index, env_step, blocks, targets, _ in states:
        if index not in selected:
            continue
        dist = distance_matrix(blocks, targets)
        success, nearest, _ = distinct_target_success(dist, threshold)
        mapping = f"b0->t{nearest[0]},b1->t{nearest[1]}"
        print(
            f"  B{index:<7d} {env_step:>6d} "
            f"{dist[0,0]:>10.6f} {dist[0,1]:>10.6f} "
            f"{dist[1,0]:>10.6f} {dist[1,1]:>10.6f} "
            f"{mapping:>12} {bool_text(success):>9}"
        )


def print_entry_retention(
    seed: int,
    outcome: str,
    branch: str,
    trace: Mapping[str, np.ndarray],
    threshold: float,
) -> None:
    events = extract_event_steps(trace)
    occupancy = boundary_occupancy_summary(trace, threshold)

    print(f"\nSeed {seed} ({outcome}) | {branch}")
    print(
        f"  {'Pair':<8} {'EventStep':>10} {'FirstBoundary':>14} "
        f"{'FirstEnv':>10} {'Occupied':>10} {'Longest':>9} "
        f"{'Final':>8} {'MinDist':>10} {'FinalDist':>11}"
    )

    for block_index in range(2):
        for target_index in range(2):
            key = f"TARGET_{block_index}_{target_index}"
            item = occupancy[(block_index, target_index)]
            event_step = events[key]
            first_boundary = item["first_boundary"]
            first_env = item["first_env_step"]
            print(
                f"  b{block_index}->t{target_index:<4d} "
                f"{str(event_step) if event_step is not None else '-':>10} "
                f"{('B'+str(first_boundary)) if first_boundary is not None else '-':>14} "
                f"{str(first_env) if first_env is not None else '-':>10} "
                f"{item['occupied_boundaries']:>10d} "
                f"{item['longest_streak']:>9d} "
                f"{bool_text(item['final_inside']):>8} "
                f"{item['minimum_distance']:>10.6f} "
                f"{item['final_distance']:>11.6f}"
            )


def print_final_success(
    seed: int,
    outcome: str,
    branch: str,
    trace: Mapping[str, np.ndarray],
    threshold: float,
) -> None:
    blocks, targets, _ = last_state(
        trace, "obs_after", np.asarray(trace["raw_action"]).shape[0] - 1
    )
    dist = distance_matrix(blocks, targets)
    success, nearest, nearest_distance = distinct_target_success(dist, threshold)
    print(
        f"{seed:<9d} {outcome:<11} {branch:<9} "
        f"b0->t{nearest[0]} ({nearest_distance[0]:.6f})  "
        f"b1->t{nearest[1]} ({nearest_distance[1]:.6f})  "
        f"distinct={bool_text(nearest[0] != nearest[1]):<3}  "
        f"success={bool_text(success)}"
    )


def collect_separation_candidates(
    seed: int,
    outcome: str,
    branch: str,
    trace: Mapping[str, np.ndarray],
    dt: float,
) -> List[Dict[str, Any]]:
    step_start = np.asarray(trace["environment_step_start"], dtype=np.int64)
    horizon = int(np.asarray(trace["raw_action"]).shape[1])
    candidates: List[Dict[str, Any]] = []

    for call in range(np.asarray(trace["raw_action"]).shape[0]):
        kin = state_kinematics(trace, call, dt)
        for block_index in range(2):
            gap_rate = float(kin["gap_rate"][block_index])
            speed = float(kin["block_speed"][block_index])
            radial_v = float(kin["radial_v"][block_index])
            gap = float(kin["gap"][block_index])
            nearest = int(kin["nearest_target"][block_index])
            distance = float(kin["distances"][block_index, nearest])

            # Ranking is only a kinematic heuristic. Positive gap rate means the
            # effector and block are separating. Block speed prevents stationary
            # far-away configurations from dominating.
            score = max(gap_rate, 0.0) * max(speed, 0.0)
            candidates.append(
                {
                    "score": score,
                    "seed": seed,
                    "outcome": outcome,
                    "branch": branch,
                    "call": call,
                    "env_end": int(step_start[call]) + horizon - 1,
                    "block": block_index,
                    "nearest_target": nearest,
                    "gap": gap,
                    "gap_rate": gap_rate,
                    "block_speed": speed,
                    "radial_v": radial_v,
                    "target_distance": distance,
                }
            )
    return candidates


def main() -> int:
    args = parse_args()
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive.")
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    root = locate_diag(args.diag)
    seeds = parse_seeds(args.seeds)
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else root / "AASF_FINAL_CAUSAL_CALLS.csv"
    )

    traces: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for seed in seeds:
        traces[seed] = {
            "original": load_trace(root, seed, "original"),
            "aasf": load_trace(root, seed, "aasf"),
        }

        o_shape = np.asarray(traces[seed]["original"]["raw_action"]).shape
        a_shape = np.asarray(traces[seed]["aasf"]["raw_action"]).shape
        if o_shape != a_shape:
            raise RuntimeError(
                f"Original/AASF action shape mismatch for seed {seed}: "
                f"{o_shape} vs {a_shape}"
            )

    print("AASF FINAL CAUSAL AUDIT")
    print("=" * 92)
    print(f"Diagnostic directory : {root}")
    print(f"Seeds                : {', '.join(map(str, seeds))}")
    print(f"Target threshold     : {args.threshold:.6f} m")
    print(f"Control interval     : {args.dt:.6f} s")
    print(
        "Interpretation note : TARGET_i_j is an exact first-entry event from the "
        "environment. Boundary retention uses only retained policy-call boundary "
        "states; chunk-internal exits cannot be reconstructed from these traces."
    )
    print(
        "Kinematic note      : gap rate and radial velocity are motion proxies, "
        "not PyBullet contact labels. GapRate > 0 means separation; RadialV < 0 "
        "means the block is approaching its current nearest target."
    )

    print("\n" + "=" * 92)
    print("CALL 0 INITIAL PERTURBATION")
    print("=" * 92)
    for seed in seeds:
        outcome = DEFAULT_CASES.get(seed, "unlabelled")
        print_call0(
            seed,
            outcome,
            traces[seed]["original"],
            traces[seed]["aasf"],
        )

    print("\n" + "=" * 92)
    print("DUAL-TARGET DISTANCES AT POLICY-CALL BOUNDARIES")
    print("=" * 92)
    for seed in seeds:
        outcome = DEFAULT_CASES.get(seed, "unlabelled")
        for branch in ("original", "aasf"):
            print_dual_target_summary(
                seed,
                outcome,
                branch,
                traces[seed][branch],
                args.threshold,
            )

    print("\n" + "=" * 92)
    print("TARGET ENTRY EVENTS AND BOUNDARY RETENTION")
    print("=" * 92)
    for seed in seeds:
        outcome = DEFAULT_CASES.get(seed, "unlabelled")
        for branch in ("original", "aasf"):
            print_entry_retention(
                seed,
                outcome,
                branch,
                traces[seed][branch],
                args.threshold,
            )

    print("\n" + "=" * 92)
    print("FINAL DISTINCT-TARGET SUCCESS MAPPING")
    print("=" * 92)
    print(
        f"{'Seed':<9} {'Outcome':<11} {'Branch':<9} "
        "nearest-target mapping and final distances"
    )
    for seed in seeds:
        outcome = DEFAULT_CASES.get(seed, "unlabelled")
        for branch in ("original", "aasf"):
            print_final_success(
                seed,
                outcome,
                branch,
                traces[seed][branch],
                args.threshold,
            )

    all_rows: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []

    for seed in seeds:
        outcome = DEFAULT_CASES.get(seed, "unlabelled")
        original = traces[seed]["original"]
        aasf = traces[seed]["aasf"]
        original_events = extract_event_steps(original)
        aasf_events = extract_event_steps(aasf)

        all_rows.extend(
            make_call_rows(
                seed,
                outcome,
                "original",
                original,
                aasf,
                args.threshold,
                args.dt,
                original_events,
            )
        )
        all_rows.extend(
            make_call_rows(
                seed,
                outcome,
                "aasf",
                aasf,
                original,
                args.threshold,
                args.dt,
                aasf_events,
            )
        )

        all_candidates.extend(
            collect_separation_candidates(
                seed, outcome, "original", original, args.dt
            )
        )
        all_candidates.extend(
            collect_separation_candidates(seed, outcome, "aasf", aasf, args.dt)
        )

    write_csv(csv_path, all_rows)

    print("\n" + "=" * 92)
    print("TOP KINEMATIC SEPARATION CANDIDATES")
    print("=" * 92)
    print(
        f"{'Rank':>4} {'Seed':>9} {'Branch':>9} {'Call':>5} {'EnvEnd':>7} "
        f"{'Block':>6} {'Near':>6} {'Gap':>10} {'GapRate':>11} "
        f"{'BlockV':>10} {'RadialV':>10} {'TargetD':>10}"
    )

    ranked = sorted(
        all_candidates,
        key=lambda item: (
            item["score"],
            item["gap_rate"],
            item["block_speed"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{rank:>4d} {item['seed']:>9d} {item['branch']:>9} "
            f"{item['call']:>5d} {item['env_end']:>7d} "
            f"{('b'+str(item['block'])):>6} "
            f"{('t'+str(item['nearest_target'])):>6} "
            f"{item['gap']:>10.6f} {item['gap_rate']:>11.6f} "
            f"{item['block_speed']:>10.6f} {item['radial_v']:>10.6f} "
            f"{item['target_distance']:>10.6f}"
        )

    print("\n" + "=" * 92)
    print("GENERATED OUTPUT")
    print("=" * 92)
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(all_rows)}")
    print("AASF FINAL CAUSAL AUDIT: COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
