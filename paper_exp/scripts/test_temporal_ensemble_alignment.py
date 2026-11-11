import numpy as np

from pathlib import Path
import sys

# Make the repository root importable when this file is
# executed directly as:
#
#   python paper_exp/scripts/test_temporal_ensemble_alignment.py
#
# Python otherwise places paper_exp/scripts at sys.path[0].
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.temporal_ensemble import (
    temporal_ensemble_current_chunk,
    append_temporal_ensemble_record,
)


def make_prediction(
    exec_start,
    slice_start,
    horizon,
    action_dim,
):
    exec_start = np.asarray(
        exec_start,
        dtype=np.int64,
    )

    batch_size = len(
        exec_start
    )

    pred = np.zeros(
        (
            batch_size,
            horizon,
            action_dim,
        ),
        dtype=np.float64,
    )

    for b in range(
        batch_size
    ):

        for j in range(
            horizon
        ):

            absolute_step = (
                int(exec_start[b])
                + j
                - int(slice_start)
            )

            pred[
                b,
                j,
            ] = np.asarray(
                [
                    10000.0 * b
                    + absolute_step
                    + 0.01 * d
                    for d in range(
                        action_dim
                    )
                ],
                dtype=np.float64,
            )

    return pred


B = 2
H = 16
A = 2
S = 3
L = 4

history = []

q0_start = np.asarray(
    [0, 0],
    dtype=np.int64,
)

q0_pred = make_prediction(
    q0_start,
    S,
    H,
    A,
)

append_temporal_ensemble_record(
    history=history,
    action_pred=q0_pred,
    slice_start=S,
    exec_start=q0_start,
    query_index=0,
)

q1_start = np.asarray(
    [4, 4],
    dtype=np.int64,
)

q1_pred = make_prediction(
    q1_start,
    S,
    H,
    A,
)

result1 = temporal_ensemble_current_chunk(
    history=history,
    current_action_pred=q1_pred,
    current_slice_start=S,
    current_exec_start=q1_start,
    current_chunk_length=L,
    beta=0.0,
)

expected1 = q1_pred[
    :,
    S:S + L,
]

if not np.allclose(
    result1["ensemble"],
    expected1,
    rtol=0.0,
    atol=0.0,
):
    raise RuntimeError(
        "Query-1 absolute-time "
        "alignment failed"
    )

if not np.array_equal(
    result1["counts"],
    np.full(
        (B, L),
        2,
        dtype=np.int64,
    ),
):
    raise RuntimeError(
        "Query-1 candidate count "
        "is incorrect"
    )

append_temporal_ensemble_record(
    history=history,
    action_pred=q1_pred,
    slice_start=S,
    exec_start=q1_start,
    query_index=1,
)

q2_start = np.asarray(
    [8, 6],
    dtype=np.int64,
)

q2_pred = make_prediction(
    q2_start,
    S,
    H,
    A,
)

result2 = temporal_ensemble_current_chunk(
    history=history,
    current_action_pred=q2_pred,
    current_slice_start=S,
    current_exec_start=q2_start,
    current_chunk_length=L,
    beta=0.25,
)

expected2 = q2_pred[
    :,
    S:S + L,
]

if not np.allclose(
    result2["ensemble"],
    expected2,
    rtol=0.0,
    atol=1e-12,
):
    raise RuntimeError(
        "Per-environment actual-step "
        "alignment failed"
    )

if not np.all(
    result2["counts"] >= 1
):
    raise RuntimeError(
        "Missing current-query prediction"
    )

old_index_b0 = (
    S
    + int(q2_start[0])
    - int(q1_start[0])
)

if old_index_b0 != 7:
    raise RuntimeError(
        "Immediate overlap index "
        "formula failed"
    )

if not np.array_equal(
    q1_pred[
        0,
        old_index_b0,
    ],
    q2_pred[
        0,
        S,
    ],
):
    raise RuntimeError(
        "Previous/current absolute-time "
        "identity failed"
    )

print(
    "TE_CURRENT_SLICE_IDENTITY = PASS"
)
print(
    "TE_PREVIOUS_QUERY_ALIGNMENT = PASS"
)
print(
    "TE_MULTIQUERY_ALIGNMENT = PASS"
)
print(
    "TE_PER_ENV_ACTUAL_CURSOR = PASS"
)
print(
    "TE_ABSOLUTE_TIME_ALIGNMENT = PASS"
)

print()
print(
    "query1 candidate counts:"
)
print(
    result1["counts"]
)

print()
print(
    "query2 candidate counts:"
)
print(
    result2["counts"]
)

print()
print(
    "TE_SYNTHETIC_ALIGNMENT_AUDIT = PASS"
)
