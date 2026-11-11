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
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
)


# ============================================================
# 0. FROZEN FORMAL SETTINGS
# ============================================================

CKPT = (
    "/root/autodl-tmp/outputs/"
    "so101_diffusion_train40_seed1000/"
    "checkpoints/020000/pretrained_model"
)

REPO = "lerobot/svla_so101_pickplace_heldout10"

DATASET_ROOT = Path(
    "/root/autodl-tmp/so101_split_40_10_v1/heldout10"
)

SPLIT_MANIFEST = Path(
    "/root/autodl-tmp/so101_split_40_10_v1/split_manifest.json"
)

DEVICE = "cuda"

OUT_DIR = Path(
    "/root/autodl-tmp/"
    "so101_heldout10_boundary_export_train40_20k_v1"
)

with open(SPLIT_MANIFEST, "r") as f:
    split_manifest = json.load(f)

ORIGINAL_EPISODES = [
    int(x)
    for x in split_manifest[
        "heldout_original_episode_indices"
    ]
]

assert ORIGINAL_EPISODES == [
    1, 3, 8, 11, 14, 19, 22, 38, 39, 47
]

BASE_QUERY_SEED = 20260825

assert torch.cuda.is_available(), (
    "CUDA required for this formal export."
)

if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
    raise RuntimeError(
        f"Output directory already contains files:\n"
        f"{OUT_DIR}\n"
        f"STOP rather than silently overwrite a formal run."
    )

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EP_DIR = OUT_DIR / "episodes"
EP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. SMALL UTILITIES
# ============================================================

def sha256_file(path, chunk_size=8 * 1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def l2(x, axis=-1):

    return np.linalg.norm(
        x,
        axis=axis,
    )


def describe(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


# ============================================================
# 2. ENVIRONMENT
# ============================================================

print("===== ENVIRONMENT =====")
print("torch version =", torch.__version__)
print("CUDA device   =", torch.cuda.get_device_name(0))
print("checkpoint    =", CKPT)
print("dataset       =", REPO)
print("dataset root  =", DATASET_ROOT)
print("original eps  =", ORIGINAL_EPISODES)
print("output        =", OUT_DIR)


# ============================================================
# 3. LOAD FROZEN POLICY
# ============================================================

print("\n===== LOAD POLICY =====")

policy = DiffusionPolicy.from_pretrained(
    CKPT
)

policy.to(DEVICE)
policy.eval()

preprocessor, postprocessor = (
    make_pre_post_processors(
        policy.config,
        pretrained_path=CKPT,
    )
)

cfg = policy.config

assert cfg.horizon == 16
assert cfg.n_obs_steps == 2
assert cfg.n_action_steps == 8

print("horizon        =", cfg.horizon)
print("n_obs_steps    =", cfg.n_obs_steps)
print("n_action_steps =", cfg.n_action_steps)
print(
    "scheduler      =",
    cfg.noise_scheduler_type,
)
print(
    "scheduler cls  =",
    type(
        policy.diffusion.noise_scheduler
    ).__name__,
)


# ============================================================
# 4. DERIVE TEMPORAL INDICES
# ============================================================

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

QUERY_STRIDE = int(
    cfg.n_action_steps
)

PREV_ALIGNED = action_offsets.index(
    QUERY_STRIDE
)

LAST_EXEC = (
    FIRST_EXEC
    + cfg.n_action_steps
    - 1
)

OVERLAP_LEN = (
    cfg.horizon
    - PREV_ALIGNED
)

print("\n===== TEMPORAL DEFINITIONS =====")
print("action offsets =", action_offsets)
print("obs offsets    =", obs_offsets)
print("FIRST_EXEC     =", FIRST_EXEC)
print("LAST_EXEC      =", LAST_EXEC)
print("Y_DATA_INDEX   =", Y_DATA_INDEX)
print("QUERY_STRIDE   =", QUERY_STRIDE)
print("PREV_ALIGNED   =", PREV_ALIGNED)
print("OVERLAP_LEN    =", OVERLAP_LEN)


# Frozen checkpoint assertions.
assert FIRST_EXEC == 1
assert LAST_EXEC == 8
assert Y_DATA_INDEX == 0
assert QUERY_STRIDE == 8
assert PREV_ALIGNED == 9
assert OVERLAP_LEN == 7

# Semantic absolute-time assertions.
assert action_offsets[FIRST_EXEC] == 0
assert action_offsets[LAST_EXEC] == 7
assert action_offsets[Y_DATA_INDEX] == -1
assert action_offsets[PREV_ALIGNED] == 8

# At a boundary between query t and query t+8:
#
# y_policy = previous[8] -> absolute t+7
# y_data   = current action offset -1 -> absolute t+7
# p        = previous[9] -> absolute t+8
# a        = current[1] -> absolute t+8
assert (
    action_offsets[LAST_EXEC]
    == QUERY_STRIDE - 1
)

assert (
    action_offsets[PREV_ALIGNED]
    == QUERY_STRIDE
)

prev_overlap_offsets = action_offsets[
    PREV_ALIGNED:
]

curr_overlap_offsets = action_offsets[
    FIRST_EXEC:
    FIRST_EXEC + OVERLAP_LEN
]

assert prev_overlap_offsets == [
    x + QUERY_STRIDE
    for x in curr_overlap_offsets
]

print("TEMPORAL SEMANTICS CHECK: PASS")


# ============================================================
# 5. LOAD REAL-ROBOT DATASET
# ============================================================

print("\n===== LOAD DATASET =====")

meta = LeRobotDatasetMetadata(
    REPO,
    root=DATASET_ROOT,
)

assert int(meta.total_episodes) == 10
assert len(ORIGINAL_EPISODES) == int(meta.total_episodes)

print("fps            =", meta.fps)
print("total episodes =", meta.total_episodes)
print("total frames   =", meta.total_frames)


def to_dt(indices):

    return [
        float(i) / float(meta.fps)
        for i in indices
    ]


delta_timestamps = {
    OBS_STATE: to_dt(
        cfg.observation_delta_indices
    ),
    ACTION: to_dt(
        cfg.action_delta_indices
    ),
}

for key in cfg.image_features:

    delta_timestamps[key] = to_dt(
        cfg.observation_delta_indices
    )


dataset = LeRobotDataset(
    REPO,
    root=DATASET_ROOT,
    delta_timestamps=delta_timestamps,
)


# ============================================================
# 6. MODEL INPUT PREPARATION
# ============================================================

def prepare_raw_batch(raw):

    raw = dict(raw)

    for key in cfg.image_features:

        x = raw[key]

        if x.dtype == torch.uint8:

            raw[key] = (
                x.float() / 255.0
            )

        elif not torch.is_floating_point(x):

            raise RuntimeError(
                f"Unexpected image dtype "
                f"for {key}: {x.dtype}"
            )

    return raw


def preprocess_to_device(raw):

    processed = preprocessor(raw)

    processed = {
        k: (
            v.to(DEVICE)
            if torch.is_tensor(v)
            else v
        )
        for k, v in processed.items()
    }

    return processed


def make_model_batch(processed):

    batch = dict(processed)

    if cfg.image_features:

        for key in cfg.image_features:

            if batch[key].ndim == 4:

                batch[key] = (
                    batch[key]
                    .unsqueeze(1)
                )

        batch[OBS_IMAGES] = torch.stack(
            [
                batch[key]
                for key in cfg.image_features
            ],
            dim=-4,
        )

    return batch


# ============================================================
# 7. FULL-HORIZON INFERENCE
#
# Query-level reproducibility uses two deterministic streams:
#
#   init_seed    -> initial diffusion x_T
#   reverse_seed -> DDPM reverse-step variance noise
#
# They are deliberately distinct.
# ============================================================

@torch.inference_mode()
def conditional_full(
    processed,
    noise,
):

    batch = make_model_batch(
        processed
    )

    batch_size, n_obs_steps = (
        batch[OBS_STATE].shape[:2]
    )

    assert n_obs_steps == cfg.n_obs_steps

    global_cond = (
        policy.diffusion
        ._prepare_global_conditioning(
            batch
        )
    )

    full = (
        policy.diffusion
        .conditional_sample(
            batch_size,
            global_cond=global_cond,
            noise=noise,
        )
    )

    return full


def seed_triplet(
    episode_index,
    query_order,
):

    logical_seed = (
        BASE_QUERY_SEED
        + int(episode_index) * 1000
        + int(query_order)
    )

    init_seed = (
        logical_seed * 2
    )

    reverse_seed = (
        logical_seed * 2 + 1
    )

    return (
        logical_seed,
        init_seed,
        reverse_seed,
    )


@torch.inference_mode()
def predict_full_seeded(
    processed,
    episode_index,
    query_order,
):

    (
        logical_seed,
        init_seed,
        reverse_seed,
    ) = seed_triplet(
        episode_index,
        query_order,
    )

    device = processed[
        OBS_STATE
    ].device

    init_gen = torch.Generator(
        device=device
    )

    init_gen.manual_seed(
        int(init_seed)
    )

    action_dim = int(
        processed[ACTION].shape[-1]
    )

    noise = torch.randn(
        (
            1,
            cfg.horizon,
            action_dim,
        ),
        device=device,
        dtype=processed[ACTION].dtype,
        generator=init_gen,
    )

    # DDPM reverse-step RNG.
    torch.manual_seed(
        int(reverse_seed)
    )

    torch.cuda.manual_seed_all(
        int(reverse_seed)
    )

    policy.reset()

    full = conditional_full(
        processed,
        noise,
    )

    return (
        full,
        noise,
        logical_seed,
        init_seed,
        reverse_seed,
    )


# ============================================================
# 8. QUERY INDEX CONSTRUCTION
#
# No temporal padding is allowed in formal export.
#
# Need:
#   observation minimum offset >= episode start
#   action maximum offset < episode end
# ============================================================

MIN_REQUIRED_OFFSET = min(
    min(obs_offsets),
    min(action_offsets),
)

MAX_REQUIRED_OFFSET = max(
    max(obs_offsets),
    max(action_offsets),
)

print("\n===== QUERY RANGE RULE =====")
print(
    "minimum required offset =",
    MIN_REQUIRED_OFFSET,
)
print(
    "maximum required offset =",
    MAX_REQUIRED_OFFSET,
)


def episode_query_indices(
    episode_index,
):

    from_idx = int(
        dataset.meta.episodes[
            "dataset_from_index"
        ][episode_index]
    )

    to_idx = int(
        dataset.meta.episodes[
            "dataset_to_index"
        ][episode_index]
    )

    # q + MIN_REQUIRED_OFFSET >= from_idx
    q_start = (
        from_idx
        - MIN_REQUIRED_OFFSET
    )

    # q + MAX_REQUIRED_OFFSET < to_idx
    # Python range stop is exclusive.
    q_stop = (
        to_idx
        - MAX_REQUIRED_OFFSET
    )

    q = list(
        range(
            q_start,
            q_stop,
            QUERY_STRIDE,
        )
    )

    assert len(q) >= 2

    for idx in q:

        assert (
            idx + MIN_REQUIRED_OFFSET
            >= from_idx
        )

        assert (
            idx + MAX_REQUIRED_OFFSET
            < to_idx
        )

    q_np = np.asarray(
        q,
        dtype=np.int64,
    )

    assert np.all(
        np.diff(q_np)
        == QUERY_STRIDE
    )

    return (
        from_idx,
        to_idx,
        q,
    )


# ============================================================
# 9. FORMAL PRE-RUN SELF-CHECK
# ============================================================

print(
    "\n===== FORMAL PRE-RUN SELF-CHECK ====="
)

(
    ep0_from,
    ep0_to,
    ep0_queries,
) = episode_query_indices(0)

first_idx = ep0_queries[0]

loader0 = DataLoader(
    Subset(
        dataset,
        [first_idx],
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

raw0 = next(iter(loader0))
raw0 = prepare_raw_batch(raw0)
proc0 = preprocess_to_device(raw0)

print(
    "episode 0 range =",
    ep0_from,
    ep0_to,
)

print(
    "first valid query =",
    first_idx,
)

for key in cfg.image_features:

    x = raw0[key]

    print(
        f"{key}: "
        f"dtype={x.dtype}, "
        f"range=["
        f"{float(x.min()):.6f}, "
        f"{float(x.max()):.6f}]"
    )

    assert float(x.min()) >= -1e-4
    assert float(x.max()) <= 1.0001


# ------------------------------------------------------------
# Query-level reproducibility test.
# ------------------------------------------------------------

(
    full_a,
    noise_a,
    logical_seed_a,
    init_seed_a,
    reverse_seed_a,
) = predict_full_seeded(
    proc0,
    0,
    0,
)

(
    full_b,
    noise_b,
    logical_seed_b,
    init_seed_b,
    reverse_seed_b,
) = predict_full_seeded(
    proc0,
    0,
    0,
)

assert logical_seed_a == logical_seed_b
assert init_seed_a == init_seed_b
assert reverse_seed_a == reverse_seed_b

assert torch.allclose(
    noise_a,
    noise_b,
    atol=0.0,
    rtol=0.0,
)

repro_max = (
    full_a - full_b
).abs().max().item()

print(
    "same-query reproducibility max diff =",
    repro_max,
)

assert torch.allclose(
    full_a,
    full_b,
    atol=1e-6,
    rtol=1e-6,
)

print(
    "QUERY REPRODUCIBILITY CHECK: PASS"
)


# ------------------------------------------------------------
# Full-horizon vs public API check, again inside formal script.
# ------------------------------------------------------------

# Recreate exactly the same initial noise.
init_gen = torch.Generator(
    device=proc0[OBS_STATE].device
)

init_gen.manual_seed(
    int(init_seed_a)
)

noise_check = torch.randn(
    (
        1,
        cfg.horizon,
        int(proc0[ACTION].shape[-1]),
    ),
    device=proc0[OBS_STATE].device,
    dtype=proc0[ACTION].dtype,
    generator=init_gen,
)

torch.manual_seed(
    int(reverse_seed_a)
)

torch.cuda.manual_seed_all(
    int(reverse_seed_a)
)

policy.reset()

with torch.inference_mode():

    public_a = (
        policy.predict_action_chunk(
            proc0,
            noise=noise_check,
        )
    )

expected_public = full_a[
    :,
    FIRST_EXEC:
    FIRST_EXEC + cfg.n_action_steps,
]

api_max = (
    public_a
    - expected_public
).abs().max().item()

print(
    "formal public-vs-full max diff =",
    api_max,
)

assert torch.allclose(
    public_a,
    expected_public,
    atol=1e-5,
    rtol=1e-5,
)

print(
    "FORMAL PUBLIC API CHECK: PASS"
)


# ============================================================
# 10. RUN ALL 50 EPISODES
# ============================================================

print(
    "\n===== BEGIN HELDOUT10 EXPORT ====="
)

all_rows = []

total_queries = 0
total_boundaries = 0


for ep in range(
    int(meta.total_episodes)
):

    (
        from_idx,
        to_idx,
        query_indices,
    ) = episode_query_indices(ep)

    loader = DataLoader(
        Subset(
            dataset,
            query_indices,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    full_norm_list = []
    full_native_list = []

    data_prev_norm_list = []
    data_prev_native_list = []

    logical_seed_list = []
    init_seed_list = []
    reverse_seed_list = []

    for query_order, (
        global_idx,
        raw,
    ) in enumerate(
        zip(
            query_indices,
            loader,
        )
    ):

        raw = prepare_raw_batch(
            raw
        )

        processed = (
            preprocess_to_device(
                raw
            )
        )

        assert (
            processed[OBS_STATE].shape[1]
            == cfg.n_obs_steps
        )

        assert (
            processed[ACTION].shape[1]
            == cfg.horizon
        )

        (
            full_norm,
            _noise,
            logical_seed,
            init_seed,
            reverse_seed,
        ) = predict_full_seeded(
            processed,
            ep,
            query_order,
        )

        assert tuple(
            full_norm.shape
        ) == (
            1,
            16,
            6,
        )

        assert torch.isfinite(
            full_norm
        ).all()

        # Convert normalized policy output back to the
        # checkpoint's native dataset action coordinates.
        full_native = postprocessor(
            full_norm
        )

        assert torch.is_tensor(
            full_native
        )

        assert tuple(
            full_native.shape
        ) == (
            1,
            16,
            6,
        )

        y_data_norm = processed[
            ACTION
        ][
            0,
            Y_DATA_INDEX,
        ]

        y_data_native = raw[
            ACTION
        ][
            0,
            Y_DATA_INDEX,
        ]

        full_norm_list.append(
            full_norm[
                0
            ].detach().cpu().numpy()
        )

        full_native_list.append(
            full_native[
                0
            ].detach().cpu().numpy()
        )

        data_prev_norm_list.append(
            y_data_norm
            .detach()
            .cpu()
            .numpy()
        )

        data_prev_native_list.append(
            y_data_native
            .detach()
            .cpu()
            .numpy()
        )

        logical_seed_list.append(
            logical_seed
        )

        init_seed_list.append(
            init_seed
        )

        reverse_seed_list.append(
            reverse_seed
        )


    full_norm = np.stack(
        full_norm_list,
        axis=0,
    )

    full_native = np.stack(
        full_native_list,
        axis=0,
    )

    data_prev_norm = np.stack(
        data_prev_norm_list,
        axis=0,
    )

    data_prev_native = np.stack(
        data_prev_native_list,
        axis=0,
    )

    query_indices_np = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    logical_seeds_np = np.asarray(
        logical_seed_list,
        dtype=np.int64,
    )

    init_seeds_np = np.asarray(
        init_seed_list,
        dtype=np.int64,
    )

    reverse_seeds_np = np.asarray(
        reverse_seed_list,
        dtype=np.int64,
    )


    # ========================================================
    # Adjacent query pairs:
    #
    # previous query = t
    # current query  = t+8
    # ========================================================

    prev_full_norm = full_norm[:-1]
    curr_full_norm = full_norm[1:]

    prev_full_native = (
        full_native[:-1]
    )

    curr_full_native = (
        full_native[1:]
    )


    # Current raw first policy action at t+8.
    a_norm = curr_full_norm[
        :,
        FIRST_EXEC,
        :,
    ]

    a_native = curr_full_native[
        :,
        FIRST_EXEC,
        :,
    ]


    # Previous time-aligned prediction for t+8.
    p_norm = prev_full_norm[
        :,
        PREV_ALIGNED,
        :,
    ]

    p_native = prev_full_native[
        :,
        PREV_ALIGNED,
        :,
    ]


    # Previous policy executable chunk's last command:
    # previous query index 8 -> absolute t+7.
    y_policy_norm = prev_full_norm[
        :,
        LAST_EXEC,
        :,
    ]

    y_policy_native = prev_full_native[
        :,
        LAST_EXEC,
        :,
    ]


    # Recorded real robot action at t+7.
    # It belongs to the current query's offset -1.
    y_data_norm = data_prev_norm[
        1:
    ]

    y_data_native = data_prev_native[
        1:
    ]


    # Same-time 7-step overlap:
    # previous [9:16] vs current [1:8].
    prev_overlap_norm = prev_full_norm[
        :,
        PREV_ALIGNED:
        PREV_ALIGNED + OVERLAP_LEN,
        :,
    ]

    curr_overlap_norm = curr_full_norm[
        :,
        FIRST_EXEC:
        FIRST_EXEC + OVERLAP_LEN,
        :,
    ]

    prev_overlap_native = (
        prev_full_native[
            :,
            PREV_ALIGNED:
            PREV_ALIGNED + OVERLAP_LEN,
            :,
        ]
    )

    curr_overlap_native = (
        curr_full_native[
            :,
            FIRST_EXEC:
            FIRST_EXEC + OVERLAP_LEN,
            :,
        ]
    )


    overlap_l2_norm = l2(
        curr_overlap_norm
        - prev_overlap_norm,
        axis=-1,
    )

    overlap_l2_native = l2(
        curr_overlap_native
        - prev_overlap_native,
        axis=-1,
    )


    query_disagreement_norm = l2(
        a_norm - p_norm
    )

    query_disagreement_native = l2(
        a_native - p_native
    )


    # Critical identity:
    # the first element of the 7-step overlap IS p vs a.
    assert np.allclose(
        query_disagreement_norm,
        overlap_l2_norm[:, 0],
        atol=1e-6,
        rtol=1e-6,
    )

    assert np.allclose(
        query_disagreement_native,
        overlap_l2_native[:, 0],
        atol=1e-6,
        rtol=1e-6,
    )


    # Two boundary references, kept separate.
    raw_jump_data_ref_norm = l2(
        a_norm - y_data_norm
    )

    raw_jump_policy_ref_norm = l2(
        a_norm - y_policy_norm
    )

    raw_jump_data_ref_native = l2(
        a_native - y_data_native
    )

    raw_jump_policy_ref_native = l2(
        a_native - y_policy_native
    )


    # How different is previous policy command from actual
    # recorded robot action at the same absolute t+7?
    prev_policy_vs_data_norm = l2(
        y_policy_norm
        - y_data_norm
    )

    prev_policy_vs_data_native = l2(
        y_policy_native
        - y_data_native
    )


    # All values must be finite.
    arrays_to_check = [
        a_norm,
        p_norm,
        y_policy_norm,
        y_data_norm,
        prev_overlap_norm,
        curr_overlap_norm,
        overlap_l2_norm,
        query_disagreement_norm,
        raw_jump_data_ref_norm,
        raw_jump_policy_ref_norm,
        prev_policy_vs_data_norm,
    ]

    for arr in arrays_to_check:

        assert np.isfinite(
            arr
        ).all()


    # Query stride check at every boundary.
    prev_query_idx = (
        query_indices_np[:-1]
    )

    curr_query_idx = (
        query_indices_np[1:]
    )

    assert np.all(
        curr_query_idx
        - prev_query_idx
        == QUERY_STRIDE
    )


    # Explicit absolute-time identities.
    assert np.all(
        prev_query_idx
        + action_offsets[LAST_EXEC]
        == curr_query_idx - 1
    )

    assert np.all(
        prev_query_idx
        + action_offsets[PREV_ALIGNED]
        == curr_query_idx
    )

    assert np.all(
        curr_query_idx
        + action_offsets[FIRST_EXEC]
        == curr_query_idx
    )


    n_queries = int(
        len(query_indices_np)
    )

    n_boundaries = int(
        n_queries - 1
    )

    total_queries += n_queries
    total_boundaries += n_boundaries


    # --------------------------------------------------------
    # Save complete episode-level trace.
    # --------------------------------------------------------

    ep_path = (
        EP_DIR
        / f"episode_{ep:02d}.npz"
    )

    np.savez_compressed(
        ep_path,

        episode_index=np.asarray(
            ep,
            dtype=np.int64,
        ),

        original_episode_index=np.asarray(
            ORIGINAL_EPISODES[ep],
            dtype=np.int64,
        ),

        episode_from_index=np.asarray(
            from_idx,
            dtype=np.int64,
        ),

        episode_to_index=np.asarray(
            to_idx,
            dtype=np.int64,
        ),

        query_indices=query_indices_np,
        logical_seeds=logical_seeds_np,
        init_seeds=init_seeds_np,
        reverse_seeds=reverse_seeds_np,

        full_norm=full_norm,
        full_native=full_native,

        data_prev_norm=data_prev_norm,
        data_prev_native=data_prev_native,

        boundary_prev_query_indices=(
            prev_query_idx
        ),

        boundary_curr_query_indices=(
            curr_query_idx
        ),

        y_data_norm=y_data_norm,
        y_policy_norm=y_policy_norm,
        a_norm=a_norm,
        p_norm=p_norm,

        y_data_native=y_data_native,
        y_policy_native=y_policy_native,
        a_native=a_native,
        p_native=p_native,

        prev_overlap_norm=(
            prev_overlap_norm
        ),

        curr_overlap_norm=(
            curr_overlap_norm
        ),

        prev_overlap_native=(
            prev_overlap_native
        ),

        curr_overlap_native=(
            curr_overlap_native
        ),

        overlap_l2_norm=(
            overlap_l2_norm
        ),

        overlap_l2_native=(
            overlap_l2_native
        ),

        query_disagreement_norm=(
            query_disagreement_norm
        ),

        query_disagreement_native=(
            query_disagreement_native
        ),

        raw_jump_data_ref_norm=(
            raw_jump_data_ref_norm
        ),

        raw_jump_policy_ref_norm=(
            raw_jump_policy_ref_norm
        ),

        raw_jump_data_ref_native=(
            raw_jump_data_ref_native
        ),

        raw_jump_policy_ref_native=(
            raw_jump_policy_ref_native
        ),

        prev_policy_vs_data_norm=(
            prev_policy_vs_data_norm
        ),

        prev_policy_vs_data_native=(
            prev_policy_vs_data_native
        ),
    )


    # --------------------------------------------------------
    # Scalar boundary table.
    # --------------------------------------------------------

    for bidx in range(
        n_boundaries
    ):

        all_rows.append({
            "episode": ep,
            "heldout_local_episode": ep,
            "original_episode": ORIGINAL_EPISODES[ep],
            "boundary_index": bidx,

            "prev_query_global_idx":
                int(prev_query_idx[bidx]),

            "curr_query_global_idx":
                int(curr_query_idx[bidx]),

            "curr_query_time_sec":
                float(
                    (
                        curr_query_idx[bidx]
                        - from_idx
                    )
                    / float(meta.fps)
                ),

            "prev_query_seed":
                int(
                    logical_seeds_np[
                        bidx
                    ]
                ),

            "curr_query_seed":
                int(
                    logical_seeds_np[
                        bidx + 1
                    ]
                ),

            "query_disagreement_norm":
                float(
                    query_disagreement_norm[
                        bidx
                    ]
                ),

            "overlap_mean_l2_norm":
                float(
                    overlap_l2_norm[
                        bidx
                    ].mean()
                ),

            "overlap_max_l2_norm":
                float(
                    overlap_l2_norm[
                        bidx
                    ].max()
                ),

            "raw_jump_data_ref_norm":
                float(
                    raw_jump_data_ref_norm[
                        bidx
                    ]
                ),

            "raw_jump_policy_ref_norm":
                float(
                    raw_jump_policy_ref_norm[
                        bidx
                    ]
                ),

            "prev_policy_vs_data_norm":
                float(
                    prev_policy_vs_data_norm[
                        bidx
                    ]
                ),

            "query_disagreement_native":
                float(
                    query_disagreement_native[
                        bidx
                    ]
                ),

            "overlap_mean_l2_native":
                float(
                    overlap_l2_native[
                        bidx
                    ].mean()
                ),

            "raw_jump_data_ref_native":
                float(
                    raw_jump_data_ref_native[
                        bidx
                    ]
                ),

            "raw_jump_policy_ref_native":
                float(
                    raw_jump_policy_ref_native[
                        bidx
                    ]
                ),

            "prev_policy_vs_data_native":
                float(
                    prev_policy_vs_data_native[
                        bidx
                    ]
                ),
        })


    print(
        f"episode {ep:02d}: "
        f"frames=[{from_idx},{to_idx}) "
        f"queries={n_queries:3d} "
        f"boundaries={n_boundaries:3d} "
        f"mean ||a-p||="
        f"{query_disagreement_norm.mean():.6f}"
    )


# ============================================================
# 11. SAVE GLOBAL BOUNDARY CSV
# ============================================================

assert len(all_rows) == total_boundaries
assert total_boundaries > 0

assert total_queries == 273, total_queries
assert total_boundaries == 263, total_boundaries

csv_path = (
    OUT_DIR
    / "boundaries.csv"
)

fieldnames = list(
    all_rows[0].keys()
)

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        all_rows
    )


# ============================================================
# 12. GLOBAL SUMMARY
# ============================================================

query_disagreement_all = np.asarray(
    [
        r[
            "query_disagreement_norm"
        ]
        for r in all_rows
    ],
    dtype=np.float64,
)

overlap_mean_all = np.asarray(
    [
        r[
            "overlap_mean_l2_norm"
        ]
        for r in all_rows
    ],
    dtype=np.float64,
)

raw_data_all = np.asarray(
    [
        r[
            "raw_jump_data_ref_norm"
        ]
        for r in all_rows
    ],
    dtype=np.float64,
)

raw_policy_all = np.asarray(
    [
        r[
            "raw_jump_policy_ref_norm"
        ]
        for r in all_rows
    ],
    dtype=np.float64,
)

policy_data_gap_all = np.asarray(
    [
        r[
            "prev_policy_vs_data_norm"
        ]
        for r in all_rows
    ],
    dtype=np.float64,
)


summary = {
    "episodes": int(meta.total_episodes),
    "queries": int(total_queries),
    "boundaries": int(total_boundaries),

    "query_disagreement_norm":
        describe(
            query_disagreement_all
        ),

    "overlap_mean_l2_norm":
        describe(
            overlap_mean_all
        ),

    "raw_jump_data_ref_norm":
        describe(
            raw_data_all
        ),

    "raw_jump_policy_ref_norm":
        describe(
            raw_policy_all
        ),

    "prev_policy_vs_data_norm":
        describe(
            policy_data_gap_all
        ),
}


# ============================================================
# 13. REPRODUCIBILITY MANIFEST
# ============================================================

model_path = (
    Path(CKPT)
    / "model.safetensors"
)

config_path = (
    Path(CKPT)
    / "config.json"
)

manifest = {
    "checkpoint": CKPT,
    "dataset": REPO,
    "dataset_root": str(DATASET_ROOT),
    "split_manifest": str(SPLIT_MANIFEST),
    "heldout_original_episode_indices": ORIGINAL_EPISODES,

    "normalization_provenance": (
        "policy preprocessor loaded from train40 checkpoint; "
        "heldout dataset statistics are not used to construct "
        "the policy preprocessor"
    ),

    "device": DEVICE,

    "model_sha256":
        sha256_file(
            model_path
        ),

    "config_sha256":
        sha256_file(
            config_path
        ),

    "torch_version":
        torch.__version__,

    "cuda_device":
        torch.cuda.get_device_name(0),

    "dataset_fps":
        float(meta.fps),

    "dataset_total_episodes":
        int(meta.total_episodes),

    "dataset_total_frames":
        int(meta.total_frames),

    "horizon":
        int(cfg.horizon),

    "n_obs_steps":
        int(cfg.n_obs_steps),

    "n_action_steps":
        int(cfg.n_action_steps),

    "observation_offsets":
        obs_offsets,

    "action_offsets":
        action_offsets,

    "first_exec_index":
        int(FIRST_EXEC),

    "last_exec_index":
        int(LAST_EXEC),

    "data_y_index":
        int(Y_DATA_INDEX),

    "query_stride":
        int(QUERY_STRIDE),

    "previous_aligned_index":
        int(PREV_ALIGNED),

    "overlap_len":
        int(OVERLAP_LEN),

    "base_query_seed":
        int(BASE_QUERY_SEED),

    "seed_rule": (
        "logical = base + episode*1000 + query_order; "
        "init_seed=2*logical; "
        "reverse_seed=2*logical+1"
    ),

    "space": (
        "full_norm and *_norm are checkpoint-native "
        "normalized action coordinates; "
        "full_native and *_native are postprocessor "
        "inverse-transformed dataset action coordinates."
    ),

    "boundary_semantics": {
        "a": (
            "current query first executable "
            "prediction at absolute t+8"
        ),
        "p": (
            "previous query time-aligned "
            "prediction for absolute t+8"
        ),
        "y_data": (
            "recorded robot action at "
            "absolute t+7"
        ),
        "y_policy": (
            "previous policy query's last "
            "executable prediction at "
            "absolute t+7"
        ),
    },

    "summary": summary,
}


manifest_path = (
    OUT_DIR
    / "manifest.json"
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


summary_path = (
    OUT_DIR
    / "summary.json"
)

with open(
    summary_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# 14. FINAL CHECKS + PRINT
# ============================================================

assert int(
    summary["episodes"]
) == 10

assert np.isfinite(
    query_disagreement_all
).all()

assert np.isfinite(
    overlap_mean_all
).all()

assert np.isfinite(
    raw_data_all
).all()

assert np.isfinite(
    raw_policy_all
).all()


print(
    "\n" + "=" * 78
)

print(
    "SO-101 HELDOUT10 BOUNDARY EXPORT COMPLETE"
)

print(
    "=" * 78
)

print(
    "episodes   =",
    total_queries and int(meta.total_episodes),
)

print(
    "queries    =",
    total_queries,
)

print(
    "boundaries =",
    total_boundaries,
)

print("\nNormalized-space descriptive statistics:")

for key, value in summary.items():

    if not isinstance(
        value,
        dict,
    ):
        continue

    print(
        f"{key:30s} "
        f"mean={value['mean']:.6f} "
        f"median={value['median']:.6f} "
        f"p95={value['p95']:.6f} "
        f"max={value['max']:.6f}"
    )

print("\nSaved:")
print(" CSV      =", csv_path)
print(" summary  =", summary_path)
print(" manifest =", manifest_path)
print(" episodes =", EP_DIR)

print(
    "\nNO BSOP PROJECTION HAS BEEN APPLIED YET."
)

print(
    "This file is the frozen held-out real-data "
    "boundary export for the next audit stage."
)

print(
    "=" * 78
)
