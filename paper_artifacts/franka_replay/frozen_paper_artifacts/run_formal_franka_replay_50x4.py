from pathlib import Path
import time
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation

ROOT = Path(
    "/root/autodl-tmp/bsop_franka_replay"
)

TRACE_ROOT = Path(
    "/root/autodl-tmp/projects/diffusion_policy/"
    "paper_artifacts/traces/"
    "bsop_aasf_confirmatory_fresh50"
)

XML = (
    ROOT
    / "mujoco_menagerie/franka_emika_panda/scene.xml"
)

OUT = ROOT / "formal_replay_v1"
OUT.mkdir(parents=True, exist_ok=True)

PER_TRAJ = OUT / "per_trajectory"
PER_TRAJ.mkdir(parents=True, exist_ok=True)

METHODS = [
    "original",
    "aasf",
    "bsop",
    "bsop_aasf",
]

ARM_DOF = 7

# ============================================================
# FROZEN PROTOCOL
# ============================================================

WORKSPACE_W = 0.20
PUSHT_CENTER = 256.0

COMMAND_DT = 0.1
PHYSICS_DT = 0.002
STEPS_PER_CMD = 50

TAU_S = 0.010

SETTLE_SECONDS = 1.0
SETTLE_STEPS = int(
    round(
        SETTLE_SECONDS / PHYSICS_DT
    )
)

POS_TOL = 1e-4
ORI_TOL = 2e-4
IK_MAX_ITER = 500
IK_DAMPING = 1e-3
IK_MAX_DQ = 0.05

DERIV_CONTEXT = 3

SAVE_TIMESERIES = True


# ============================================================
# IK model
# ============================================================

ik_model = mujoco.MjModel.from_xml_path(
    str(XML)
)
ik_data = mujoco.MjData(ik_model)

if not np.isclose(
    ik_model.opt.timestep,
    PHYSICS_DT
):
    raise RuntimeError(
        f"Unexpected physics dt "
        f"{ik_model.opt.timestep}"
    )

home_id = mujoco.mj_name2id(
    ik_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

hand_id = mujoco.mj_name2id(
    ik_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

mujoco.mj_resetDataKeyframe(
    ik_model,
    ik_data,
    home_id
)
mujoco.mj_forward(
    ik_model,
    ik_data
)

q_home = (
    ik_data.qpos[:ARM_DOF]
    .copy()
)

hand_center = (
    ik_data.xpos[hand_id]
    .copy()
)

R_home = (
    ik_data.xmat[hand_id]
    .reshape(3, 3)
    .copy()
)

q_lo = (
    ik_model.actuator_ctrlrange[
        :ARM_DOF, 0
    ]
    .copy()
)

q_hi = (
    ik_model.actuator_ctrlrange[
        :ARM_DOF, 1
    ]
    .copy()
)


def pusht_to_xyz(action):

    action = np.asarray(
        action,
        dtype=np.float64
    )

    xyz = hand_center.copy()

    xyz[0] += (
        WORKSPACE_W
        * (action[0] - PUSHT_CENTER)
        / 512.0
    )

    xyz[1] += (
        WORKSPACE_W
        * (action[1] - PUSHT_CENTER)
        / 512.0
    )

    return xyz


def set_ik_q(q):

    ik_data.qpos[:ARM_DOF] = q
    ik_data.qvel[:] = 0.0

    mujoco.mj_forward(
        ik_model,
        ik_data
    )


def get_pose_errors(target_pos):

    current_pos = (
        ik_data.xpos[hand_id]
        .copy()
    )

    current_R = (
        ik_data.xmat[hand_id]
        .reshape(3, 3)
        .copy()
    )

    pos_err = (
        target_pos - current_pos
    )

    R_err = (
        R_home @ current_R.T
    )

    ori_err = (
        Rotation
        .from_matrix(R_err)
        .as_rotvec()
    )

    return pos_err, ori_err


def solve_pose_ik(
    target_pos,
    q_init
):

    q = np.asarray(
        q_init,
        dtype=np.float64
    ).copy()

    set_ik_q(q)

    for it in range(IK_MAX_ITER):

        pos_err, ori_err = (
            get_pose_errors(
                target_pos
            )
        )

        pe = np.linalg.norm(
            pos_err
        )

        oe = np.linalg.norm(
            ori_err
        )

        if (
            pe < POS_TOL
            and oe < ORI_TOL
        ):
            return (
                True,
                q.copy(),
                pe,
                oe,
                it + 1,
            )

        jacp = np.zeros(
            (3, ik_model.nv)
        )

        jacr = np.zeros(
            (3, ik_model.nv)
        )

        mujoco.mj_jacBody(
            ik_model,
            ik_data,
            jacp,
            jacr,
            hand_id,
        )

        J = np.vstack([
            jacp[:, :ARM_DOF],
            jacr[:, :ARM_DOF],
        ])

        e = np.concatenate([
            pos_err,
            ori_err,
        ])

        A = (
            J @ J.T
            + IK_DAMPING ** 2
            * np.eye(6)
        )

        dq = (
            J.T
            @ np.linalg.solve(
                A,
                e
            )
        )

        dn = np.linalg.norm(dq)

        if dn > IK_MAX_DQ:
            dq *= IK_MAX_DQ / dn

        q += dq

        q = np.clip(
            q,
            q_lo,
            q_hi
        )

        set_ik_q(q)

    pe_vec, oe_vec = (
        get_pose_errors(
            target_pos
        )
    )

    return (
        False,
        q.copy(),
        np.linalg.norm(pe_vec),
        np.linalg.norm(oe_vec),
        IK_MAX_ITER,
    )


def build_ik(commands):

    targets = np.stack([
        pusht_to_xyz(a)
        for a in commands
    ])

    q_des = []

    pe_all = []
    oe_all = []
    it_all = []
    margin_all = []

    q_prev = q_home.copy()

    for i, target in enumerate(
        targets
    ):

        (
            ok,
            q,
            pe,
            oe,
            nit,
        ) = solve_pose_ik(
            target,
            q_prev
        )

        if not ok:
            raise RuntimeError(
                f"IK failed at "
                f"command {i}; "
                f"pe={pe}, oe={oe}"
            )

        margin = np.min(
            np.minimum(
                q - q_lo,
                q_hi - q
            )
        )

        q_des.append(q)
        pe_all.append(pe)
        oe_all.append(oe)
        it_all.append(nit)
        margin_all.append(margin)

        q_prev = q

    return (
        targets,
        np.asarray(q_des),
        np.asarray(pe_all),
        np.asarray(oe_all),
        np.asarray(it_all),
        np.asarray(margin_all),
    )


def mean_diff_norm(
    x,
    order
):

    d = np.diff(
        x,
        n=order,
        axis=0
    )

    return float(
        np.linalg.norm(
            d,
            axis=1
        ).mean()
    )


def replay_one(
    seed,
    method,
    commands
):

    (
        targets,
        q_des,
        ik_pe,
        ik_oe,
        ik_it,
        margins,
    ) = build_ik(commands)

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(model)

    h_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    ee_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand"
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        h_id
    )

    # Start exactly at first desired pose
    data.qpos[:ARM_DOF] = q_des[0]
    data.qvel[:] = 0.0

    data.ctrl[:] = (
        model.key_ctrl[h_id]
    )

    q_ref = q_des[0].copy()

    data.ctrl[:ARM_DOF] = q_ref

    mujoco.mj_forward(
        model,
        data
    )

    alpha = (
        1.0
        - np.exp(
            -PHYSICS_DT
            / TAU_S
        )
    )

    # ========================================================
    # settle -- excluded
    # ========================================================

    for _ in range(
        SETTLE_STEPS
    ):

        data.ctrl[
            :ARM_DOF
        ] = q_ref

        mujoco.mj_step(
            model,
            data
        )

    settle_tracking = (
        np.linalg.norm(
            data.xpos[ee_id]
            - targets[0]
        )
    )

    # Context for finite differences
    context_pos = []
    context_tau = []

    for _ in range(
        DERIV_CONTEXT
    ):

        data.ctrl[
            :ARM_DOF
        ] = q_ref

        mujoco.mj_step(
            model,
            data
        )

        context_pos.append(
            data.xpos[ee_id]
            .copy()
        )

        context_tau.append(
            data.qfrc_actuator[
                :ARM_DOF
            ].copy()
        )

    # ========================================================
    # formal dynamics
    # ========================================================

    hand_log = []
    target_log = []
    tau_log = []
    force_log = []
    tracking_log = []
    command_index_log = []

    for cmd_i in range(
        1,
        len(q_des)
    ):

        q_target = q_des[cmd_i]
        xyz_target = targets[cmd_i]

        for _ in range(
            STEPS_PER_CMD
        ):

            q_ref = (
                q_ref
                + alpha
                * (
                    q_target
                    - q_ref
                )
            )

            data.ctrl[
                :ARM_DOF
            ] = q_ref

            mujoco.mj_step(
                model,
                data
            )

            hand_xyz = (
                data.xpos[ee_id]
                .copy()
            )

            hand_log.append(
                hand_xyz
            )

            target_log.append(
                xyz_target
            )

            tau_log.append(
                data.qfrc_actuator[
                    :ARM_DOF
                ].copy()
            )

            force_log.append(
                data.actuator_force[
                    :ARM_DOF
                ].copy()
            )

            tracking_log.append(
                np.linalg.norm(
                    hand_xyz
                    - xyz_target
                )
            )

            command_index_log.append(
                cmd_i
            )

    hand = np.asarray(
        hand_log
    )

    target_log = np.asarray(
        target_log
    )

    tau = np.asarray(
        tau_log
    )

    force = np.asarray(
        force_log
    )

    tracking = np.asarray(
        tracking_log
    )

    command_index_log = np.asarray(
        command_index_log
    )

    context_pos = np.asarray(
        context_pos
    )

    context_tau = np.asarray(
        context_tau
    )

    n = len(hand)

    # ========================================================
    # derivatives
    # ========================================================

    all_tau = np.vstack([
        context_tau,
        tau
    ])

    tau_rate_all = (
        np.diff(
            all_tau,
            axis=0
        )
        / PHYSICS_DT
    )

    tau_rate = (
        tau_rate_all[-n:]
    )

    tau_rate_norm = (
        np.linalg.norm(
            tau_rate,
            axis=1
        )
    )

    all_pos = np.vstack([
        context_pos,
        hand
    ])

    ee_jerk = (
        np.diff(
            all_pos,
            n=3,
            axis=0
        )
        / PHYSICS_DT ** 3
    )

    if len(ee_jerk) != n:
        raise RuntimeError(
            "jerk alignment failed"
        )

    ee_jerk_norm = (
        np.linalg.norm(
            ee_jerk,
            axis=1
        )
    )

    # ========================================================
    # saturation
    # ========================================================

    force_limit = np.max(
        np.abs(
            model.actuator_forcerange[
                :ARM_DOF
            ]
        ),
        axis=1
    )

    sat = (
        np.abs(force)
        >= 0.99
        * force_limit[None, :]
    )

    any_sat = np.any(
        sat,
        axis=1
    )

    joint_sat_pct = (
        100.0
        * np.mean(
            sat,
            axis=0
        )
    )

    # ========================================================
    # command metrics
    # ========================================================

    D1 = mean_diff_norm(
        targets,
        1
    )

    D2 = mean_diff_norm(
        targets,
        2
    )

    D3 = mean_diff_norm(
        targets,
        3
    )

    qstep = np.diff(
        q_des,
        axis=0
    )

    # ========================================================
    # summary
    # ========================================================

    row = {
        "seed": seed,
        "method": method,

        "common_commands":
            len(commands),

        "formal_commands":
            len(commands) - 1,

        "formal_duration_s":
            n * PHYSICS_DT,

        "command_D1_m":
            D1,

        "command_D2_m":
            D2,

        "command_D3_m":
            D3,

        "tracking_rmse_mm":
            1000.0
            * np.sqrt(
                np.mean(
                    tracking ** 2
                )
            ),

        "tracking_p95_mm":
            1000.0
            * np.percentile(
                tracking,
                95
            ),

        "torque_rate_rms_Nm_s":
            np.sqrt(
                np.mean(
                    tau_rate_norm ** 2
                )
            ),

        "torque_rate_p95_Nm_s":
            np.percentile(
                tau_rate_norm,
                95
            ),

        "ee_jerk_rms_m_s3":
            np.sqrt(
                np.mean(
                    ee_jerk_norm ** 2
                )
            ),

        "ee_jerk_p95_m_s3":
            np.percentile(
                ee_jerk_norm,
                95
            ),

        "any_joint_saturation_pct":
            100.0
            * np.mean(
                any_sat
            ),

        "max_single_joint_saturation_pct":
            float(
                np.max(
                    joint_sat_pct
                )
            ),

        "J7_saturation_pct":
            float(
                joint_sat_pct[6]
            ),

        "settle_tracking_mm":
            1000.0
            * settle_tracking,

        "ik_max_pos_error_m":
            float(
                np.max(ik_pe)
            ),

        "ik_max_ori_error_rad":
            float(
                np.max(ik_oe)
            ),

        "ik_min_joint_margin_rad":
            float(
                np.min(
                    margins
                )
            ),

        "ik_max_iterations":
            int(
                np.max(
                    ik_it
                )
            ),

        "ik_max_q_step_norm_rad":
            float(
                np.max(
                    np.linalg.norm(
                        qstep,
                        axis=1
                    )
                )
            ),

        "actuator_force_vs_qfrc_max_abs":
            float(
                np.max(
                    np.abs(
                        force - tau
                    )
                )
            ),
    }

    # Optional raw replay archive
    if SAVE_TIMESERIES:

        np.savez_compressed(
            PER_TRAJ
            / (
                f"{method}_seed"
                f"{seed}_timeseries.npz"
            ),

            target_xyz=target_log,
            hand_xyz=hand,
            q_des=q_des,

            qfrc_actuator=tau,
            actuator_force=force,

            torque_rate=tau_rate,
            torque_rate_norm=tau_rate_norm,

            ee_jerk=ee_jerk,
            ee_jerk_norm=ee_jerk_norm,

            tracking_error_m=tracking,

            saturation=sat,
            command_index=command_index_log,

            target_commands_xyz=targets,
        )

    return row


# ============================================================
# Formal 50 x 4
# ============================================================

rows = []

start_all = time.time()

for seed in range(
    4300100,
    4300150
):

    traces = {}

    for method in METHODS:

        path = (
            TRACE_ROOT
            / method
            / f"test_seed_{seed}.npz"
        )

        z = np.load(
            path,
            allow_pickle=True
        )

        x = np.asarray(
            z["executed_action"],
            dtype=np.float64
        )

        if len(x) != int(
            z[
                "actual_executed_steps"
            ]
        ):
            raise RuntimeError(
                f"{seed} {method}: "
                "execution accounting mismatch"
            )

        traces[method] = x

    common_T = min(
        len(x)
        for x in traces.values()
    )

    # Strong fairness check:
    # all methods begin from same command.
    first = np.stack([
        traces[m][0]
        for m in METHODS
    ])

    if np.max(
        np.abs(
            first - first[0]
        )
    ) > 1e-6:

        raise RuntimeError(
            f"{seed}: first action mismatch"
        )

    print(
        f"\n========== SEED {seed} "
        f"| common T={common_T} =========="
    )

    for method in METHODS:

        commands = (
            traces[method][
                :common_T
            ]
        )

        t0 = time.time()

        row = replay_one(
            seed,
            method,
            commands
        )

        rows.append(row)

        elapsed = (
            time.time() - t0
        )

        print(
            f"{method:10s}"
            f" | torque-rate "
            f"{row['torque_rate_rms_Nm_s']:.1f}"
            f" | jerk "
            f"{row['ee_jerk_rms_m_s3']:.1f}"
            f" | track "
            f"{row['tracking_rmse_mm']:.2f} mm"
            f" | sat "
            f"{row['any_joint_saturation_pct']:.3f}%"
            f" | {elapsed:.1f}s"
        )

        # Incremental save so interrupted jobs resume safely.
        temp_df = pd.DataFrame(
            rows
        )

        temp_df.to_csv(
            OUT
            / "formal_replay_partial.csv",
            index=False
        )


df = pd.DataFrame(rows)

out_csv = (
    OUT
    / "formal_replay_per_seed.csv"
)

df.to_csv(
    out_csv,
    index=False
)

elapsed_all = (
    time.time() - start_all
)

print("\n===== FORMAL REPLAY COMPLETE =====")

print("rows:", len(df))
print("expected:", 50 * 4)
print(
    "total elapsed [min]:",
    elapsed_all / 60.0
)

print("\nMean health metrics:")

print(
    df.groupby("method")[
        [
            "tracking_rmse_mm",
            "any_joint_saturation_pct",
            "torque_rate_rms_Nm_s",
            "ee_jerk_rms_m_s3",
        ]
    ].mean()
)

print("\nSaved:", out_csv)

print(
    "\nPASS: frozen 50x4 Franka "
    "formal replay completed."
)
