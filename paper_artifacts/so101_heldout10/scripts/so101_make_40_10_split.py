import hashlib
import json
from pathlib import Path

import numpy as np

from lerobot.datasets import (
    LeRobotDataset,
    split_dataset,
)


# ============================================================
# 0. FROZEN SPLIT DEFINITION
# ============================================================

SOURCE_REPO = "lerobot/svla_so101_pickplace"

OUT_ROOT = Path(
    "/root/autodl-tmp/so101_split_40_10_v1"
)

SPLIT_SEED = 20260825

# Generated once from an episode-level random split and now frozen.
HELDOUT_ORIGINAL = [
    1, 3, 8, 11, 14,
    19, 22, 38, 39, 47,
]

TRAIN_ORIGINAL = [
    i
    for i in range(50)
    if i not in HELDOUT_ORIGINAL
]

assert len(TRAIN_ORIGINAL) == 40
assert len(HELDOUT_ORIGINAL) == 10
assert set(TRAIN_ORIGINAL).isdisjoint(
    HELDOUT_ORIGINAL
)
assert sorted(
    TRAIN_ORIGINAL + HELDOUT_ORIGINAL
) == list(range(50))


# Frozen BSOP parameters.
ETA = 1.0
LAMBDA_MAX = 0.5
KAPPA = 0.5
RHO_RMS_PER_DIM = 0.10
ACTION_DIM = 6

RHO_NORM_L2 = (
    RHO_RMS_PER_DIM
    * np.sqrt(ACTION_DIM)
)


# ============================================================
# 1. SAFETY
# ============================================================

if OUT_ROOT.exists():
    raise RuntimeError(
        f"{OUT_ROOT} already exists.\n"
        "STOP rather than overwrite the frozen split."
    )


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

print("=" * 78)
print("SO-101 40/10 HELD-OUT SPLIT")
print("=" * 78)

print("source repo =", SOURCE_REPO)
print("split seed  =", SPLIT_SEED)

source = LeRobotDataset(
    SOURCE_REPO
)

assert source.meta.total_episodes == 50

print(
    "source episodes =",
    source.meta.total_episodes,
)

print(
    "source frames   =",
    source.meta.total_frames,
)

print()
print(
    "train original episodes   =",
    TRAIN_ORIGINAL,
)

print(
    "heldout original episodes =",
    HELDOUT_ORIGINAL,
)


# ============================================================
# 3. CREATE PHYSICALLY SEPARATE DATASETS
# ============================================================

splits = split_dataset(
    source,
    splits={
        "train40": TRAIN_ORIGINAL,
        "heldout10": HELDOUT_ORIGINAL,
    },
    output_dir=OUT_ROOT,
)

train_ds = splits["train40"]
heldout_ds = splits["heldout10"]

assert train_ds.meta.total_episodes == 40
assert heldout_ds.meta.total_episodes == 10


# ============================================================
# 4. BASIC DATA INTEGRITY
# ============================================================

print()
print("===== TRAIN40 =====")

print(
    "repo id  =",
    train_ds.repo_id,
)

print(
    "root     =",
    train_ds.root,
)

print(
    "episodes =",
    train_ds.meta.total_episodes,
)

print(
    "frames   =",
    train_ds.meta.total_frames,
)

sample_train = train_ds[0]

assert tuple(
    sample_train["action"].shape
) == (6,)

assert tuple(
    sample_train["observation.state"].shape
) == (6,)

print(
    "sample action shape =",
    tuple(
        sample_train["action"].shape
    ),
)

print()
print("===== HELDOUT10 =====")

print(
    "repo id  =",
    heldout_ds.repo_id,
)

print(
    "root     =",
    heldout_ds.root,
)

print(
    "episodes =",
    heldout_ds.meta.total_episodes,
)

print(
    "frames   =",
    heldout_ds.meta.total_frames,
)

sample_test = heldout_ds[0]

assert tuple(
    sample_test["action"].shape
) == (6,)

assert tuple(
    sample_test["observation.state"].shape
) == (6,)

print(
    "sample action shape =",
    tuple(
        sample_test["action"].shape
    ),
)


# ============================================================
# 5. VERIFY SPLIT-SPECIFIC ACTION STATISTICS
# ============================================================

def as_list(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).tolist()


print()
print("===== ACTION NORMALIZATION STATS =====")

for name, ds in [
    ("train40", train_ds),
    ("heldout10", heldout_ds),
]:

    stats = ds.meta.stats["action"]

    print()
    print(name)

    for key in [
        "min",
        "max",
        "mean",
        "std",
    ]:
        if key in stats:
            print(
                f"{key:6s} =",
                as_list(stats[key]),
            )


# ============================================================
# 6. HASH SPLIT METADATA
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(
                8 * 1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


hashes = {}

for name, ds in [
    ("train40", train_ds),
    ("heldout10", heldout_ds),
]:

    for rel in [
        "meta/info.json",
        "meta/stats.json",
    ]:

        path = (
            Path(ds.root)
            / rel
        )

        assert path.exists(), path

        hashes[
            f"{name}/{rel}"
        ] = sha256_file(path)


# ============================================================
# 7. FREEZE MANIFEST
# ============================================================

manifest = {
    "experiment":
        "SO-101 40/10 training-held-out hardening",
    "source_repo":
        SOURCE_REPO,
    "split_seed":
        SPLIT_SEED,
    "train_original_episode_indices":
        TRAIN_ORIGINAL,
    "heldout_original_episode_indices":
        HELDOUT_ORIGINAL,
    "train_local_repo_id":
        train_ds.repo_id,
    "heldout_local_repo_id":
        heldout_ds.repo_id,
    "train_root":
        str(train_ds.root),
    "heldout_root":
        str(heldout_ds.root),
    "train_episodes":
        int(train_ds.meta.total_episodes),
    "heldout_episodes":
        int(heldout_ds.meta.total_episodes),
    "train_frames":
        int(train_ds.meta.total_frames),
    "heldout_frames":
        int(heldout_ds.meta.total_frames),

    "future_training_protocol": {
        "policy": "diffusion",
        "seed": 1000,
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "batch_size": 4,
        "steps": 20000,
    },

    "frozen_bsop": {
        "eta": ETA,
        "lambda_max": LAMBDA_MAX,
        "kappa": KAPPA,
        "rho_rms_per_dim":
            RHO_RMS_PER_DIM,
        "rho_norm_l2":
            float(RHO_NORM_L2),
    },

    "metadata_sha256":
        hashes,

    "important_note":
        (
            "Held-out episodes are excluded from "
            "the new policy training and from its "
            "normalization statistics. BSOP "
            "parameters are frozen before this "
            "held-out retraining/evaluation."
        ),
}

manifest_path = (
    OUT_ROOT
    / "split_manifest.json"
)

with open(
    manifest_path,
    "w",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# 8. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("SO-101 40/10 SPLIT COMPLETE")
print("=" * 78)

print(
    "train episodes   =",
    train_ds.meta.total_episodes,
)

print(
    "heldout episodes =",
    heldout_ds.meta.total_episodes,
)

print(
    "train frames     =",
    train_ds.meta.total_frames,
)

print(
    "heldout frames   =",
    heldout_ds.meta.total_frames,
)

print()
print(
    "train root   =",
    train_ds.root,
)

print(
    "heldout root =",
    heldout_ds.root,
)

print()
print(
    "rho_norm_l2 =",
    RHO_NORM_L2,
)

print()
print("Metadata hashes:")

for k, v in hashes.items():
    print(k, v)

print()
print(
    "manifest =",
    manifest_path,
)

print()
print(
    "TRAIN / HELDOUT EPISODE OVERLAP: 0"
)

print(
    "SPLIT-SPECIFIC STATS: CREATED"
)

print(
    "FROZEN BSOP PARAMETERS: RECORDED"
)

print("=" * 78)
