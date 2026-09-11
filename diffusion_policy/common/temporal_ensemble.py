import numpy as np


def temporal_ensemble_current_chunk(
    history,
    current_action_pred,
    current_slice_start,
    current_exec_start,
    current_chunk_length,
    beta=0.0,
):
    """
    Temporal ensemble aligned by absolute actually-executed
    control-step index.

    For query q:

        action_pred[:, slice_start + k]

    corresponds to absolute control step:

        exec_start + k

    Historical predictions are combined only when they refer
    to exactly the same absolute executed control step.
    """

    current_action_pred = np.asarray(
        current_action_pred
    )

    if current_action_pred.ndim != 3:
        raise RuntimeError(
            "current_action_pred must have shape "
            "(batch, horizon, action_dim)"
        )

    batch_size, horizon, action_dim = (
        current_action_pred.shape
    )

    current_exec_start = np.asarray(
        current_exec_start,
        dtype=np.int64,
    )

    if current_exec_start.shape != (
        batch_size,
    ):
        raise RuntimeError(
            "current_exec_start must have shape "
            f"({batch_size},), got "
            f"{current_exec_start.shape}"
        )

    current_slice_start = int(
        current_slice_start
    )
    current_chunk_length = int(
        current_chunk_length
    )
    beta = float(beta)

    if beta < 0:
        raise RuntimeError(
            "beta must be non-negative"
        )

    current_slice_end = (
        current_slice_start
        + current_chunk_length
    )

    if not (
        0 <= current_slice_start
        and current_slice_end <= horizon
    ):
        raise RuntimeError(
            "Current execution slice is outside "
            "action_pred: "
            f"slice=[{current_slice_start}:"
            f"{current_slice_end}], "
            f"horizon={horizon}"
        )

    records = []

    for record in history:
        pred = np.asarray(
            record["action_pred"]
        )

        if pred.ndim != 3:
            raise RuntimeError(
                "Historical action_pred must be 3D"
            )

        if pred.shape[0] != batch_size:
            raise RuntimeError(
                "Historical batch size differs"
            )

        if pred.shape[2] != action_dim:
            raise RuntimeError(
                "Historical action dimension differs"
            )

        exec_start = np.asarray(
            record["exec_start"],
            dtype=np.int64,
        )

        if exec_start.shape != (
            batch_size,
        ):
            raise RuntimeError(
                "Historical exec_start has "
                "wrong shape"
            )

        records.append(
            dict(
                action_pred=pred,
                slice_start=int(
                    record["slice_start"]
                ),
                exec_start=exec_start,
                query_index=int(
                    record["query_index"]
                ),
            )
        )

    if records:
        current_query_index = (
            max(
                r["query_index"]
                for r in records
            )
            + 1
        )
    else:
        current_query_index = 0

    records.append(
        dict(
            action_pred=current_action_pred,
            slice_start=current_slice_start,
            exec_start=current_exec_start.copy(),
            query_index=current_query_index,
        )
    )

    ensemble = np.zeros(
        (
            batch_size,
            current_chunk_length,
            action_dim,
        ),
        dtype=np.float64,
    )

    counts = np.zeros(
        (
            batch_size,
            current_chunk_length,
        ),
        dtype=np.int64,
    )

    weight_sums = np.zeros(
        (
            batch_size,
            current_chunk_length,
        ),
        dtype=np.float64,
    )

    source_age_min = np.full(
        (
            batch_size,
            current_chunk_length,
        ),
        -1,
        dtype=np.int64,
    )

    source_age_max = np.full(
        (
            batch_size,
            current_chunk_length,
        ),
        -1,
        dtype=np.int64,
    )

    for b in range(batch_size):

        for k in range(
            current_chunk_length
        ):

            absolute_step = (
                int(current_exec_start[b])
                + k
            )

            weighted_sum = np.zeros(
                (action_dim,),
                dtype=np.float64,
            )

            ages = []

            for record in records:

                relative_step = (
                    absolute_step
                    - int(
                        record[
                            "exec_start"
                        ][b]
                    )
                )

                pred_index = (
                    int(
                        record[
                            "slice_start"
                        ]
                    )
                    + relative_step
                )

                pred_horizon = int(
                    record[
                        "action_pred"
                    ].shape[1]
                )

                if not (
                    0
                    <= pred_index
                    < pred_horizon
                ):
                    continue

                age = (
                    current_query_index
                    - int(
                        record[
                            "query_index"
                        ]
                    )
                )

                if age < 0:
                    raise RuntimeError(
                        "Negative query age"
                    )

                weight = float(
                    np.exp(
                        -beta * age
                    )
                )

                value = np.asarray(
                    record[
                        "action_pred"
                    ][
                        b,
                        pred_index,
                    ],
                    dtype=np.float64,
                )

                weighted_sum += (
                    weight * value
                )

                weight_sums[
                    b,
                    k,
                ] += weight

                counts[
                    b,
                    k,
                ] += 1

                ages.append(
                    age
                )

            if counts[b, k] == 0:
                raise RuntimeError(
                    "No aligned prediction "
                    f"for batch={b}, k={k}"
                )

            ensemble[
                b,
                k,
            ] = (
                weighted_sum
                / weight_sums[
                    b,
                    k,
                ]
            )

            source_age_min[
                b,
                k,
            ] = min(
                ages
            )

            source_age_max[
                b,
                k,
            ] = max(
                ages
            )

    return dict(
        ensemble=ensemble,
        counts=counts,
        weight_sums=weight_sums,
        source_age_min=source_age_min,
        source_age_max=source_age_max,
        current_query_index=(
            current_query_index
        ),
    )


def append_temporal_ensemble_record(
    history,
    action_pred,
    slice_start,
    exec_start,
    query_index,
    max_history=None,
):
    """
    Append one policy-query prediction record.

    exec_start must come from actual executed control-step
    accounting, never from planned chunk count.
    """

    record = dict(
        action_pred=np.asarray(
            action_pred
        ).copy(),
        slice_start=int(
            slice_start
        ),
        exec_start=np.asarray(
            exec_start,
            dtype=np.int64,
        ).copy(),
        query_index=int(
            query_index
        ),
    )

    history.append(
        record
    )

    if max_history is not None:

        max_history = int(
            max_history
        )

        if max_history <= 0:
            raise RuntimeError(
                "max_history must be positive"
            )

        if len(history) > max_history:
            del history[
                :-max_history
            ]

    return history
