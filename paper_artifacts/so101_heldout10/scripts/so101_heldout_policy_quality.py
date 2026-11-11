import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION


# ============================================================
# 0. FROZEN SETTINGS
# ============================================================

CKPT = Path(
    "/root/autodl-tmp/outputs/"
    "so101_diffusion_train40_seed1000/"
    "checkpoints/020000/pretrained_model"
)

RAW = Path(
    "/root/autodl-tmp/"
    "so101_heldout10_boundary_export_train40_20k_v1"
)

DATASET_ROOT = Path(
    "/root/autodl-tmp/"
    "so101_split_40_10_v1/heldout10"
)

REPO = "lerobot/svla_so101_pickplace_heldout10"

OUT = Path(
    "/root/autodl-tmp/"
    "so101_heldout10_policy_quality_train40_20k_v1"
)

EXPECTED_MODEL_SHA256 = (
    "c0001423c9a09fd4baf3f639654cc1320"
    "c6d152828e516f2b9f7c75c83ce6df4"
)

EXPECTED_CONFIG_SHA256 = (
    "f88b0539406cd500249c76f0f70aa950"
    "8068ca8e57bdcff22577b0b070ceac96"
)

EXPECTED_EPISODES = 10
EXPECTED_QUERIES = 273
EXPECTED_ACTION_DIM = 6


# ============================================================
# 1. UTILITIES
# ============================================================

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def metrics_from_error(err):
    err = np.asarray(err, dtype=np.float64)

    assert err.ndim == 2
    assert err.shape[1] == EXPECTED_ACTION_DIM
    assert np.isfinite(err).all()

    l2 = np.linalg.norm(err, axis=1)

    return {
        "n_queries": int(err.shape[0]),
        "coordinate_rmse": float(
            np.sqrt(np.mean(err ** 2))
        ),
        "coordinate_mae": float(
            np.mean(np.abs(err))
        ),
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "l2_p95": float(
            np.percentile(l2, 95)
        ),
        "l2_max": float(np.max(l2)),
    }


def per_dim_metrics(err):
    err = np.asarray(err, dtype=np.float64)

    return [
        {
            "dimension": int(d),
            "rmse": float(
                np.sqrt(
                    np.mean(err[:, d] ** 2)
                )
            ),
            "mae": float(
                np.mean(
                    np.abs(err[:, d])
                )
            ),
        }
        for d in range(err.shape[1])
    ]


# ============================================================
# 2. PROVENANCE
# ============================================================

print("===== SO-101 HELDOUT POLICY-QUALITY DIAGNOSTIC =====")

assert CKPT.is_dir()
assert RAW.is_dir()
assert DATASET_ROOT.is_dir()

if OUT.exists() and any(OUT.iterdir()):
    raise RuntimeError(
        "Formal output directory already contains files:\n"
        f"{OUT}\n"
        "STOP rather than overwrite."
    )

model_sha = sha256_file(
    CKPT / "model.safetensors"
)

config_sha = sha256_file(
    CKPT / "config.json"
)

print("checkpoint =", CKPT)
print("raw export =", RAW)
print("dataset    =", DATASET_ROOT)
print("model sha  =", model_sha)
print("config sha =", config_sha)

assert model_sha == EXPECTED_MODEL_SHA256
assert config_sha == EXPECTED_CONFIG_SHA256

print("FROZEN CHECKPOINT HASH CHECK: PASS")


# ============================================================
# 3. LOAD CONFIG + CHECKPOINT PREPROCESSOR
#
# No policy inference is performed.
# Loading the policy here is only a robust way to recover the
# exact frozen config used to construct its preprocessor.
# ============================================================

print()
print("===== LOAD FROZEN CONFIG / PREPROCESSOR =====")

policy = DiffusionPolicy.from_pretrained(
    str(CKPT)
)

cfg = policy.config

preprocessor, _postprocessor = (
    make_pre_post_processors(
        cfg,
        pretrained_path=str(CKPT),
    )
)

action_offsets = [
    int(x)
    for x in cfg.action_delta_indices
]

obs_offsets = [
    int(x)
    for x in cfg.observation_delta_indices
]

FIRST_EXEC = action_offsets.index(0)
Y_DATA_INDEX = action_offsets.index(-1)

assert cfg.horizon == 16
assert cfg.n_obs_steps == 2
assert cfg.n_action_steps == 8
assert FIRST_EXEC == 1
assert Y_DATA_INDEX == 0
assert action_offsets == [
    -1, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14,
]

print("action offsets =", action_offsets)
print("FIRST_EXEC     =", FIRST_EXEC)
print("Y_DATA_INDEX   =", Y_DATA_INDEX)
print("PREPROCESSOR CONFIG CHECK: PASS")

# We do not need model weights after this point.
del policy


# ============================================================
# 4. LOAD EXACT HELDOUT DATASET WITH SAME TEMPORAL INDEXING
# ============================================================

print()
print("===== LOAD HELDOUT10 DATASET =====")

meta = LeRobotDatasetMetadata(
    REPO,
    root=DATASET_ROOT,
)

assert int(meta.total_episodes) == EXPECTED_EPISODES

delta_timestamps = resolve_delta_timestamps(
    cfg,
    meta,
)

assert delta_timestamps is not None
assert ACTION in delta_timestamps
assert len(delta_timestamps[ACTION]) == len(action_offsets)

print(
    "delta timestamp keys =",
    sorted(delta_timestamps.keys()),
)
print(
    "action delta ts      =",
    delta_timestamps[ACTION],
)

dataset = LeRobotDataset(
    REPO,
    root=DATASET_ROOT,
    delta_timestamps=delta_timestamps,
)

print("fps            =", meta.fps)
print("total episodes =", meta.total_episodes)
print("total frames   =", meta.total_frames)

assert int(meta.total_episodes) == 10


# ============================================================
# 5. RAW BATCH PREPARATION
# ============================================================

def prepare_raw_batch(raw):
    raw = dict(raw)

    for key in cfg.image_features:
        if key not in raw:
            continue

        x = raw[key]

        if x.dtype == torch.uint8:
            raw[key] = x.float() / 255.0

        elif not torch.is_floating_point(x):
            raise RuntimeError(
                f"Unexpected image dtype "
                f"for {key}: {x.dtype}"
            )

    return raw


# ============================================================
# 6. EXACT QUERY-LEVEL RECONSTRUCTION
# ============================================================

print()
print("===== RECONSTRUCT SAME-TIME RECORDED ACTIONS =====")

ep_files = sorted(
    (RAW / "episodes").glob(
        "episode_*.npz"
    )
)

assert len(ep_files) == EXPECTED_EPISODES

all_pred_norm = []
all_data_norm = []
all_error_norm = []

per_query_rows = []
per_episode_rows = []

max_prev_norm_alignment_error = 0.0
max_prev_native_alignment_error = 0.0

total_queries = 0


for ep, ep_path in enumerate(ep_files):

    with np.load(
        ep_path,
        allow_pickle=False,
    ) as z:

        query_indices = np.asarray(
            z["query_indices"],
            dtype=np.int64,
        )

        full_norm = np.asarray(
            z["full_norm"],
            dtype=np.float64,
        )

        full_native = np.asarray(
            z["full_native"],
            dtype=np.float64,
        )

        saved_prev_norm = np.asarray(
            z["data_prev_norm"],
            dtype=np.float64,
        )

        saved_prev_native = np.asarray(
            z["data_prev_native"],
            dtype=np.float64,
        )

        original_episode_index = int(
            np.asarray(
                z["original_episode_index"]
            ).item()
        )

        saved_from = int(
            np.asarray(
                z["episode_from_index"]
            ).item()
        )

        saved_to = int(
            np.asarray(
                z["episode_to_index"]
            ).item()
        )


    n_q = len(query_indices)

    assert full_norm.shape == (
        n_q,
        16,
        EXPECTED_ACTION_DIM,
    )

    assert full_native.shape == (
        n_q,
        16,
        EXPECTED_ACTION_DIM,
    )

    assert saved_prev_norm.shape == (
        n_q,
        EXPECTED_ACTION_DIM,
    )

    assert saved_prev_native.shape == (
        n_q,
        EXPECTED_ACTION_DIM,
    )


    # Cross-check episode boundaries against current local copy.
    meta_from = int(
        dataset.meta.episodes[
            "dataset_from_index"
        ][ep]
    )

    meta_to = int(
        dataset.meta.episodes[
            "dataset_to_index"
        ][ep]
    )

    assert saved_from == meta_from
    assert saved_to == meta_to


    loader = DataLoader(
        Subset(
            dataset,
            query_indices.tolist(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


    ep_pred = []
    ep_data = []


    for q_order, (
        global_idx,
        raw,
    ) in enumerate(
        zip(
            query_indices,
            loader,
        )
    ):

        raw = prepare_raw_batch(raw)

        processed = preprocessor(raw)

        # ----------------------------------------------------
        # ALIGNMENT AUDIT:
        # offset -1 must exactly reconstruct the action that
        # was already frozen in the original boundary export.
        # ----------------------------------------------------

        reconstructed_prev_norm = (
            processed[ACTION][
                0,
                Y_DATA_INDEX,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        reconstructed_prev_native = (
            raw[ACTION][
                0,
                Y_DATA_INDEX,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        dn = float(
            np.max(
                np.abs(
                    reconstructed_prev_norm
                    - saved_prev_norm[q_order]
                )
            )
        )

        dx = float(
            np.max(
                np.abs(
                    reconstructed_prev_native
                    - saved_prev_native[q_order]
                )
            )
        )

        max_prev_norm_alignment_error = max(
            max_prev_norm_alignment_error,
            dn,
        )

        max_prev_native_alignment_error = max(
            max_prev_native_alignment_error,
            dx,
        )

        assert dn <= 1e-6, (
            ep,
            q_order,
            global_idx,
            dn,
        )

        assert dx <= 1e-6, (
            ep,
            q_order,
            global_idx,
            dx,
        )


        # ----------------------------------------------------
        # FORMAL POLICY-QUALITY PAIR
        #
        # prediction:
        #   frozen policy prediction at action offset 0
        #
        # target:
        #   recorded real-robot action at action offset 0
        #
        # Both are in checkpoint-normalized action space.
        # ----------------------------------------------------

        pred_norm = full_norm[
            q_order,
            FIRST_EXEC,
            :,
        ]

        data_norm = (
            processed[ACTION][
                0,
                FIRST_EXEC,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        pred_native = full_native[
            q_order,
            FIRST_EXEC,
            :,
        ]

        data_native = (
            raw[ACTION][
                0,
                FIRST_EXEC,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        assert np.isfinite(pred_norm).all()
        assert np.isfinite(data_norm).all()

        err_norm = pred_norm - data_norm
        l2_norm = float(
            np.linalg.norm(err_norm)
        )


        ep_pred.append(pred_norm)
        ep_data.append(data_norm)

        row = {
            "episode_index": int(ep),
            "original_episode_index":
                original_episode_index,
            "query_order": int(q_order),
            "query_global_index":
                int(global_idx),
            "l2_error_norm": l2_norm,
        }

        for d in range(
            EXPECTED_ACTION_DIM
        ):
            row[f"pred_norm_d{d}"] = float(
                pred_norm[d]
            )
            row[f"data_norm_d{d}"] = float(
                data_norm[d]
            )
            row[f"error_norm_d{d}"] = float(
                err_norm[d]
            )

            # Native values are archived for audit only.
            row[f"pred_native_d{d}"] = float(
                pred_native[d]
            )
            row[f"data_native_d{d}"] = float(
                data_native[d]
            )

        per_query_rows.append(row)


    ep_pred = np.stack(
        ep_pred,
        axis=0,
    )

    ep_data = np.stack(
        ep_data,
        axis=0,
    )

    ep_err = ep_pred - ep_data

    ep_metrics = metrics_from_error(
        ep_err
    )

    ep_row = {
        "episode_index": int(ep),
        "original_episode_index":
            int(original_episode_index),
        **ep_metrics,
    }

    per_episode_rows.append(
        ep_row
    )

    all_pred_norm.append(ep_pred)
    all_data_norm.append(ep_data)
    all_error_norm.append(ep_err)

    total_queries += n_q

    print(
        f"episode {ep:02d}: "
        f"original={original_episode_index:2d} "
        f"queries={n_q:2d} "
        f"RMSE={ep_metrics['coordinate_rmse']:.6f} "
        f"MAE={ep_metrics['coordinate_mae']:.6f} "
        f"medianL2={ep_metrics['l2_median']:.6f} "
        f"P95L2={ep_metrics['l2_p95']:.6f}"
    )


# ============================================================
# 7. FINAL ALIGNMENT ASSERTIONS
# ============================================================

print()
print("===== ALIGNMENT AUDIT =====")

print(
    "max prev normalized mismatch =",
    f"{max_prev_norm_alignment_error:.12g}",
)

print(
    "max prev native mismatch     =",
    f"{max_prev_native_alignment_error:.12g}",
)

assert max_prev_norm_alignment_error <= 1e-6
assert max_prev_native_alignment_error <= 1e-6

assert total_queries == EXPECTED_QUERIES
assert len(per_query_rows) == EXPECTED_QUERIES

print("QUERY / DATASET / PREPROCESSOR ALIGNMENT: PASS")


# ============================================================
# 8. POOLED FORMAL METRICS
# ============================================================

pred_all = np.concatenate(
    all_pred_norm,
    axis=0,
)

data_all = np.concatenate(
    all_data_norm,
    axis=0,
)

err_all = np.concatenate(
    all_error_norm,
    axis=0,
)

assert pred_all.shape == (
    EXPECTED_QUERIES,
    EXPECTED_ACTION_DIM,
)

assert data_all.shape == (
    EXPECTED_QUERIES,
    EXPECTED_ACTION_DIM,
)

pooled = metrics_from_error(
    err_all
)

dim_metrics = per_dim_metrics(
    err_all
)


# ============================================================
# 9. SAVE FORMAL ARTIFACTS
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


per_query_path = (
    OUT / "policy_quality_per_query.csv"
)

with open(
    per_query_path,
    "w",
    newline="",
) as f:

    fieldnames = list(
        per_query_rows[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        per_query_rows
    )


per_episode_path = (
    OUT / "policy_quality_per_episode.csv"
)

with open(
    per_episode_path,
    "w",
    newline="",
) as f:

    fieldnames = list(
        per_episode_rows[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        per_episode_rows
    )


summary = {
    "protocol": (
        "Frozen 20k train40 Diffusion Policy predictions "
        "versus same-time recorded heldout10 actions; "
        "no policy resampling."
    ),
    "space": (
        "checkpoint-normalized action space"
    ),
    "prediction_definition": (
        "full_norm[query, action_offset_0]"
    ),
    "target_definition": (
        "checkpoint preprocessor applied to recorded "
        "heldout action at action_offset_0"
    ),
    "episodes": EXPECTED_EPISODES,
    "queries": EXPECTED_QUERIES,
    "action_dim": EXPECTED_ACTION_DIM,
    "pooled": pooled,
    "per_dimension": dim_metrics,
    "alignment_audit": {
        "max_prev_norm_abs_mismatch":
            max_prev_norm_alignment_error,
        "max_prev_native_abs_mismatch":
            max_prev_native_alignment_error,
        "status": "PASS",
    },
}

summary_path = (
    OUT / "summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


manifest = {
    "checkpoint": str(CKPT),
    "raw_export": str(RAW),
    "dataset_root": str(DATASET_ROOT),
    "repo_id": REPO,
    "model_sha256": model_sha,
    "config_sha256": config_sha,
    "episodes": EXPECTED_EPISODES,
    "queries": EXPECTED_QUERIES,
    "action_dim": EXPECTED_ACTION_DIM,
    "action_offsets": action_offsets,
    "first_exec_index": FIRST_EXEC,
    "y_data_index": Y_DATA_INDEX,
    "normalization_source": (
        "train40 20k checkpoint policy preprocessor"
    ),
    "policy_resampled": False,
    "outputs": {
        "per_query": str(
            per_query_path
        ),
        "per_episode": str(
            per_episode_path
        ),
        "summary": str(
            summary_path
        ),
    },
}

manifest_path = (
    OUT / "manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
)


# ============================================================
# 10. REPORT
# ============================================================

print()
print("=" * 78)
print("SO-101 HELDOUT POLICY-QUALITY DIAGNOSTIC COMPLETE")
print("=" * 78)

print("episodes             =", EXPECTED_EPISODES)
print("queries              =", EXPECTED_QUERIES)
print("action dimensions    =", EXPECTED_ACTION_DIM)

print()
print("----- POOLED NORMALIZED FIRST-ACTION ERROR -----")
print(
    "coordinate RMSE      =",
    f"{pooled['coordinate_rmse']:.6f}",
)
print(
    "coordinate MAE       =",
    f"{pooled['coordinate_mae']:.6f}",
)
print(
    "mean 6-D L2          =",
    f"{pooled['l2_mean']:.6f}",
)
print(
    "median 6-D L2        =",
    f"{pooled['l2_median']:.6f}",
)
print(
    "P95 6-D L2           =",
    f"{pooled['l2_p95']:.6f}",
)
print(
    "max 6-D L2           =",
    f"{pooled['l2_max']:.6f}",
)

print()
print("----- PER-DIMENSION NORMALIZED ERROR -----")

for m in dim_metrics:
    print(
        f"d{m['dimension']}: "
        f"RMSE={m['rmse']:.6f} "
        f"MAE={m['mae']:.6f}"
    )

print()
print("Saved:")
print(" per-query   =", per_query_path)
print(" per-episode =", per_episode_path)
print(" summary     =", summary_path)
print(" manifest    =", manifest_path)

print()
print("NO POLICY INFERENCE OR RESAMPLING WAS PERFORMED.")
print("POLICY-QUALITY DIAGNOSTIC: PASS")
print("=" * 78)
