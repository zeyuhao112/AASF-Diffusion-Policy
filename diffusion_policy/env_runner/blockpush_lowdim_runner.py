import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gym.wrappers import FlattenObservation

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class BlockPushLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=5,
            crf=22,
            past_action=False,
            abs_action=False,
            obs_eef_target=True,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 10
        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    FlattenObservation(
                        BlockPushMultimodal(
                            control_frequency=task_fps,
                            shared_memory=False,
                            seed=seed,
                            abs_action=abs_action
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, shared_memory=False)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_eef_target = obs_eef_target


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # BEGIN BLOCKPUSH PAIRED POLICY SEED CONFIG
        paired_policy_seed_enabled = (
            os.environ.get(
                "DP_PAIRED_POLICY_SEED",
                "0",
            ) == "1"
        )

        policy_seed_salt = int(
            os.environ.get(
                "DP_POLICY_SEED_SALT",
                "20260710",
            )
        )
        # END BLOCKPUSH PAIRED POLICY SEED CONFIG

        # BEGIN VERIFIED BLOCKPUSH BSOP CONFIG
        bsop_enabled = (
            os.environ.get(
                "DP_BSOP_ENABLE",
                "0",
            ) == "1"
        )

        bsop_eta = float(
            os.environ.get(
                "DP_BSOP_ETA",
                "1.0",
            )
        )

        bsop_lambda_max = float(
            os.environ.get(
                "DP_BSOP_LAMBDA_MAX",
                "0.5",
            )
        )

        # Cross-task scale transfer:
        # rho/tau = 0.5.
        bsop_rho = float(
            os.environ.get(
                "DP_BSOP_RHO",
                "0.009907585256047251",
            )
        )

        bsop_kappa = float(
            os.environ.get(
                "DP_BSOP_KAPPA",
                "0.5",
            )
        )

        if not (0.0 <= bsop_eta <= 1.0):
            raise ValueError(
                "DP_BSOP_ETA must lie in [0, 1]."
            )

        if not (
            0.0 <= bsop_lambda_max <= 1.0
        ):
            raise ValueError(
                "DP_BSOP_LAMBDA_MAX must lie in [0, 1]."
            )

        if bsop_rho < 0.0:
            raise ValueError(
                "DP_BSOP_RHO must be non-negative."
            )

        if not (
            0.0 <= bsop_kappa <= 1.0
        ):
            raise ValueError(
                "DP_BSOP_KAPPA must lie in [0, 1]."
            )
        # END VERIFIED BLOCKPUSH BSOP CONFIG

        # plan for rollout
        n_envs = len(self.env_fns)

        if bsop_enabled and n_envs != 1:
            raise RuntimeError(
                "BlockPush BSOP evaluation requires "
                "n_envs=1."
            )

        if paired_policy_seed_enabled and n_envs != 1:
            raise RuntimeError(
                "DP_PAIRED_POLICY_SEED requires "
                "DP_N_ENVS=1."
            )
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            # BEGIN BLOCKPUSH PAIRED POLICY SEED
            #
            # With n_envs=1, every chunk is one episode.
            # Seed the diffusion noise stream from the
            # environment seed so different methods receive
            # identical policy randomness for the same episode.
            episode_policy_seed = None

            if paired_policy_seed_enabled:
                episode_env_seed = int(
                    self.env_seeds[start]
                )

                episode_policy_seed = (
                    episode_env_seed * 1000003
                    + policy_seed_salt
                ) % 2147483647

                torch.manual_seed(
                    episode_policy_seed
                )

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        episode_policy_seed
                    )
            # END BLOCKPUSH PAIRED POLICY SEED

            # BEGIN VERIFIED BLOCKPUSH BSOP STATE
            bsop_previous_action_pred = None
            bsop_previous_executed_last = None
            bsop_previous_action_slice_start = None
            bsop_previous_executed_chunk_length = None
            # END VERIFIED BLOCKPUSH BSOP STATE

            # BEGIN AASF FORENSIC TRACE: chunk initialization
            trace_enabled = (
                os.environ.get("DP_AASF_TRACE", "0") == "1"
            )
            trace_seed_spec = os.environ.get(
                "DP_AASF_TRACE_SEEDS", ""
            )
            trace_target_seeds = {
                int(item.strip())
                for item in trace_seed_spec.split(",")
                if item.strip()
            }
            trace_chunk_seeds = [
                int(self.env_seeds[start + local_index])
                for local_index in range(this_n_active_envs)
            ]
            trace_this_chunk = (
                trace_enabled
                and (
                    not trace_target_seeds
                    or any(
                        seed in trace_target_seeds
                        for seed in trace_chunk_seeds
                    )
                )
            )
            trace_records = [
                []
                for _ in range(this_n_active_envs)
            ]
            trace_action_offset = 0

            # BEGIN BLOCKPUSH ACTUAL EXECUTION ACCOUNTING
            #
            # MultiStepWrapper.reward contains one entry
            # for every action that was truly executed by
            # the wrapped environment. We use its cumulative
            # length as a read-only execution counter.
            trace_cumulative_executed_steps = np.zeros(
                (n_envs,),
                dtype=np.int64,
            )

            trace_planned_steps_per_query = [
                []
                for _ in range(this_n_active_envs)
            ]

            trace_executed_steps_per_query = [
                []
                for _ in range(this_n_active_envs)
            ]

            trace_actual_executed_action_parts = [
                []
                for _ in range(this_n_active_envs)
            ]
            # END BLOCKPUSH ACTUAL EXECUTION ACCOUNTING

            # END AASF FORENSIC TRACE: chunk initialization

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # BEGIN AASF FORENSIC TRACE: pre-policy observation
                trace_obs_before = (
                    obs.copy()
                    if trace_this_chunk
                    else None
                )
                # END AASF FORENSIC TRACE: pre-policy observation

                # create obs dict
                if not self.obs_eef_target:
                    obs[...,8:10] = 0
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # Original policy command before any
                # inference-time post-processing.
                raw_action = action.copy()

                # Boundary-Safe Overlap Projection.
                #
                # y: previous chunk's final executed action
                # a: current chunk's first raw action
                # p: previous query's prediction aligned to a
                #
                # b = a + lambda * (p - a)
                #
                # lambda is restricted to [0, lambda_star],
                # where lambda_star minimizes ||b-y|| on the
                # line segment between a and p. Therefore the
                # projected boundary jump cannot exceed the
                # unprojected jump, apart from floating-point
                # tolerance.
                bsop_current_action_pred = None
                bsop_current_action_slice_start = None
                bsop_current_executed_chunk_length = None

                # BEGIN BLOCKPUSH BSOP DIAGNOSTICS
                #
                # Read-only mechanism records. These values
                # must never participate in action generation.
                bsop_diag_available = np.zeros(
                    (action.shape[0],),
                    dtype=bool,
                )
                bsop_diag_raw_boundary_jump = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_projected_boundary_jump = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_correction = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_relative_correction = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_lambda_star = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_lambda = np.full(
                    (action.shape[0],),
                    np.nan,
                    dtype=np.float64,
                )
                bsop_diag_active = np.zeros(
                    (action.shape[0],),
                    dtype=bool,
                )
                # END BLOCKPUSH BSOP DIAGNOSTICS

                if bsop_enabled:
                    if "action_pred" not in np_action_dict:
                        raise RuntimeError(
                            "BSOP requires policy output "
                            "'action_pred'."
                        )

                    bsop_current_action_pred = np.asarray(
                        np_action_dict["action_pred"]
                    )

                    if bsop_current_action_pred.ndim != 3:
                        raise RuntimeError(
                            "BSOP expected action_pred shape "
                            "(batch, horizon, action_dim), got "
                            f"{bsop_current_action_pred.shape}."
                        )

                    bsop_pred_steps_only = bool(
                        getattr(
                            policy,
                            "pred_action_steps_only",
                            False,
                        )
                    )
                    bsop_n_obs_steps = int(
                        getattr(
                            policy,
                            "n_obs_steps",
                            self.n_obs_steps,
                        )
                    )
                    bsop_oa_convention = bool(
                        getattr(
                            policy,
                            "oa_step_convention",
                            False,
                        )
                    )

                    if bsop_pred_steps_only:
                        bsop_policy_slice_start = 0
                    else:
                        bsop_policy_slice_start = (
                            bsop_n_obs_steps
                            - int(bsop_oa_convention)
                        )

                    bsop_current_action_slice_start = (
                        bsop_policy_slice_start
                        + int(getattr(self, "n_latency_steps", 0))
                    )
                    bsop_current_executed_chunk_length = int(
                        action.shape[1]
                    )

                    bsop_slice_end = (
                        bsop_current_action_slice_start
                        + bsop_current_executed_chunk_length
                    )

                    bsop_expected_raw_action = (
                        bsop_current_action_pred[
                            :,
                            bsop_current_action_slice_start:
                            bsop_slice_end,
                        ]
                    )

                    if (
                        bsop_expected_raw_action.shape
                        != raw_action.shape
                    ):
                        raise RuntimeError(
                            "BSOP inferred action slice has "
                            "the wrong shape: "
                            f"action_pred="
                            f"{bsop_current_action_pred.shape}, "
                            f"slice=["
                            f"{bsop_current_action_slice_start}:"
                            f"{bsop_slice_end}], "
                            f"expected="
                            f"{bsop_expected_raw_action.shape}, "
                            f"returned={raw_action.shape}."
                        )

                    if not np.array_equal(
                        bsop_expected_raw_action,
                        raw_action,
                    ):
                        bsop_max_diff = float(
                            np.max(
                                np.abs(
                                    bsop_expected_raw_action
                                    - raw_action
                                )
                            )
                        )
                        raise RuntimeError(
                            "BSOP action_pred alignment "
                            "check failed. "
                            f"max_abs_diff="
                            f"{bsop_max_diff:.9g}"
                        )

                    if (
                        bsop_previous_action_pred is not None
                    ):
                        bsop_previous_overlap_index = (
                            int(
                                bsop_previous_action_slice_start
                            )
                            + int(
                                bsop_previous_executed_chunk_length
                            )
                        )

                        if not (
                            0
                            <= bsop_previous_overlap_index
                            < bsop_previous_action_pred.shape[1]
                        ):
                            raise RuntimeError(
                                "BSOP previous overlap index "
                                "is outside action_pred: "
                                f"index="
                                f"{bsop_previous_overlap_index}, "
                                f"horizon="
                                f"{bsop_previous_action_pred.shape[1]}."
                            )

                        # Use float64 for projection geometry.
                        bsop_y = np.asarray(
                            bsop_previous_executed_last,
                            dtype=np.float64,
                        )
                        bsop_a = np.asarray(
                            action[:, 0],
                            dtype=np.float64,
                        )
                        bsop_p = np.asarray(
                            bsop_previous_action_pred[
                                :,
                                bsop_previous_overlap_index,
                            ],
                            dtype=np.float64,
                        )

                        if not (
                            bsop_y.shape
                            == bsop_a.shape
                            == bsop_p.shape
                        ):
                            raise RuntimeError(
                                "BSOP vector shapes differ: "
                                f"y={bsop_y.shape}, "
                                f"a={bsop_a.shape}, "
                                f"p={bsop_p.shape}."
                            )

                        bsop_d = bsop_a - bsop_y
                        bsop_q = bsop_p - bsop_a

                        bsop_q_norm_sq = np.sum(
                            bsop_q * bsop_q,
                            axis=-1,
                        )
                        bsop_q_norm = np.sqrt(
                            bsop_q_norm_sq
                        )
                        bsop_raw_jump = np.linalg.norm(
                            bsop_d,
                            axis=-1,
                        )

                        bsop_lambda_star = np.zeros_like(
                            bsop_q_norm_sq
                        )

                        bsop_valid = (
                            bsop_q_norm_sq > 1.0e-12
                        )

                        bsop_lambda_star[bsop_valid] = (
                            -np.sum(
                                bsop_d[bsop_valid]
                                * bsop_q[bsop_valid],
                                axis=-1,
                            )
                            / bsop_q_norm_sq[bsop_valid]
                        )

                        bsop_lambda_star = np.clip(
                            bsop_lambda_star,
                            0.0,
                            1.0,
                        )

                        bsop_lambda = (
                            bsop_eta
                            * bsop_lambda_star
                        )
                        bsop_lambda = np.minimum(
                            bsop_lambda,
                            bsop_lambda_max,
                        )
                        bsop_lambda = np.minimum(
                            bsop_lambda,
                            bsop_rho
                            / (
                                bsop_q_norm
                                + 1.0e-12
                            ),
                        )

                        # Relative response protection:
                        #
                        # ||b-a|| <= kappa * ||a-y||
                        #
                        # This limits how much BSOP may alter
                        # the latest policy command relative
                        # to the original boundary movement.
                        bsop_lambda = np.minimum(
                            bsop_lambda,
                            bsop_kappa
                            * bsop_raw_jump
                            / (
                                bsop_q_norm
                                + 1.0e-12
                            ),
                        )
                        bsop_lambda = np.maximum(
                            bsop_lambda,
                            0.0,
                        )

                        bsop_projected_first = (
                            bsop_a
                            + bsop_lambda[:, None]
                            * bsop_q
                        )

                        bsop_baseline_jump = (
                            bsop_raw_jump
                        )
                        bsop_projected_jump = np.linalg.norm(
                            bsop_projected_first - bsop_y,
                            axis=-1,
                        )
                        bsop_correction = np.linalg.norm(
                            bsop_projected_first - bsop_a,
                            axis=-1,
                        )

                        if np.any(
                            bsop_projected_jump
                            > bsop_baseline_jump
                            + 1.0e-8
                        ):
                            raise RuntimeError(
                                "BSOP boundary guarantee "
                                "failed before dtype cast."
                            )

                        if np.any(
                            bsop_correction
                            > bsop_rho
                            + 1.0e-8
                        ):
                            raise RuntimeError(
                                "BSOP correction-radius "
                                "constraint failed."
                            )

                        if np.any(
                            bsop_correction
                            > (
                                bsop_kappa
                                * bsop_baseline_jump
                                + 1.0e-8
                            )
                        ):
                            raise RuntimeError(
                                "BSOP relative-response "
                                "constraint failed."
                            )

                        action = action.copy()
                        action[:, 0] = (
                            bsop_projected_first.astype(
                                action.dtype,
                                copy=False,
                            )
                        )

                        # Recheck after float32 assignment.
                        bsop_cast_jump = np.linalg.norm(
                            np.asarray(
                                action[:, 0],
                                dtype=np.float64,
                            )
                            - bsop_y,
                            axis=-1,
                        )
                        bsop_cast_correction = np.linalg.norm(
                            np.asarray(
                                action[:, 0],
                                dtype=np.float64,
                            )
                            - bsop_a,
                            axis=-1,
                        )

                        # BEGIN BLOCKPUSH BSOP DIAGNOSTICS
                        #
                        # Record the actual float32 command
                        # that will subsequently enter AASF /
                        # env.step. No control value is changed.
                        bsop_diag_available[:] = True

                        bsop_diag_raw_boundary_jump[:] = (
                            bsop_baseline_jump
                        )
                        bsop_diag_projected_boundary_jump[:] = (
                            bsop_cast_jump
                        )
                        bsop_diag_correction[:] = (
                            bsop_cast_correction
                        )

                        bsop_diag_relative_correction[:] = (
                            bsop_cast_correction
                            / (
                                bsop_baseline_jump
                                + 1.0e-12
                            )
                        )

                        bsop_diag_lambda_star[:] = (
                            bsop_lambda_star
                        )
                        bsop_diag_lambda[:] = (
                            bsop_lambda
                        )

                        bsop_diag_active[:] = (
                            bsop_cast_correction
                            > 1.0e-12
                        )
                        # END BLOCKPUSH BSOP DIAGNOSTICS

                        if np.any(
                            bsop_cast_jump
                            > bsop_baseline_jump
                            + 1.0e-4
                        ):
                            raise RuntimeError(
                                "BSOP boundary guarantee "
                                "failed after dtype cast."
                            )

                        if np.any(
                            bsop_cast_correction
                            > bsop_rho
                            + 1.0e-4
                        ):
                            raise RuntimeError(
                                "BSOP correction cap failed "
                                "after dtype cast."
                            )

                        if np.any(
                            bsop_cast_correction
                            > (
                                bsop_kappa
                                * bsop_baseline_jump
                                + 1.0e-4
                            )
                        ):
                            raise RuntimeError(
                                "BSOP relative-response cap "
                                "failed after dtype cast."
                            )

                # BEGIN SCALE-AWARE AASF (BlockPush)
                action_smooth_mode = os.environ.get("DP_ACTION_SMOOTH_MODE", "off").strip().lower()

                if action_smooth_mode not in {"off", "fixed", "adaptive"}:
                    raise ValueError(
                        "DP_ACTION_SMOOTH_MODE must be "
                        "'off', 'fixed', or 'adaptive'."
                    )

                if action_smooth_mode != "off":
                    action_smooth_alpha = float(
                        os.environ.get("DP_ACTION_SMOOTH_ALPHA", "0.8")
                    )
                    alpha_min = float(os.environ.get(
                        "DP_ACTION_SMOOTH_ALPHA_MIN", "0.45"
                    ))
                    alpha_max = float(os.environ.get(
                        "DP_ACTION_SMOOTH_ALPHA_MAX",
                        str(action_smooth_alpha),
                    ))
                    gain = float(os.environ.get(
                        "DP_ACTION_SMOOTH_GAIN", "0.25"
                    ))

                    if not (
                        0.0 < alpha_min
                        <= alpha_max
                        <= 1.0
                    ):
                        raise ValueError(
                            "AASF alpha values must satisfy "
                            "0 < alpha_min <= alpha_max <= 1."
                        )

                    if not (0.0 < action_smooth_alpha <= 1.0):
                        raise ValueError(
                            "DP_ACTION_SMOOTH_ALPHA must be "
                            "in (0, 1]."
                        )

                    if gain < 0.0:
                        raise ValueError(
                            "DP_ACTION_SMOOTH_GAIN must be "
                            "non-negative."
                        )

                    threshold = None
                    threshold_source = "not-used"

                    if action_smooth_mode == "adaptive":
                        threshold_override = os.environ.get(
                            "DP_ACTION_SMOOTH_TAU"
                        )

                        if threshold_override not in (None, ""):
                            threshold = float(threshold_override)
                            threshold_source = "override"
                        else:
                            threshold_ratio = float(
                                os.environ.get(
                                    "DP_ACTION_THRESHOLD_RATIO",
                                    "1.2403473459",
                                )
                            )
                            task_median_step = float(
                                os.environ.get(
                                    "DP_ACTION_MEDIAN_STEP_NORM",
                                    "nan",
                                )
                            )

                            if (
                                not np.isfinite(task_median_step)
                                or task_median_step <= 0.0
                            ):
                                raise RuntimeError(
                                    "Scale-aware AASF requires "
                                    "a positive "
                                    "DP_ACTION_MEDIAN_STEP_NORM."
                                )

                            threshold = (
                                threshold_ratio
                                * task_median_step
                            )
                            threshold_source = "ratio-times-median"

                        if (
                            not np.isfinite(threshold)
                            or threshold <= 0.0
                        ):
                            raise ValueError(
                                "The AASF threshold must be "
                                "positive and finite."
                            )

                    raw_action = action.copy()

                    if raw_action.ndim != 3:
                        raise RuntimeError(
                            "Expected action shape [batch, time, dim], "
                            f"received {raw_action.shape}."
                        )

                    smoothed_action = raw_action.copy()
                    trigger_count = 0
                    transition_count = 0

                    for action_index in range(
                        1, smoothed_action.shape[1]
                    ):
                        if action_smooth_mode == "adaptive":
                            action_delta = np.linalg.norm(
                                raw_action[:, action_index]
                                - smoothed_action[
                                    :, action_index - 1
                                ],
                                axis=-1,
                                keepdims=True,
                            )

                            trigger_count += int(
                                np.count_nonzero(
                                    action_delta > threshold
                                )
                            )
                            transition_count += int(
                                action_delta.size
                            )

                            excess = np.maximum(
                                action_delta - threshold,
                                0.0,
                            ) / (threshold + 1e-12)

                            adaptive_alpha = (
                                action_smooth_alpha
                                - gain * excess
                            )
                            adaptive_alpha = np.clip(
                                adaptive_alpha,
                                alpha_min,
                                alpha_max,
                            )
                        else:
                            adaptive_alpha = action_smooth_alpha

                        smoothed_action[:, action_index] = (
                            adaptive_alpha
                            * raw_action[:, action_index]
                            + (1.0 - adaptive_alpha)
                            * smoothed_action[
                                :, action_index - 1
                            ]
                        )

                    action = smoothed_action.astype(
                        raw_action.dtype,
                        copy=False,
                    )

                    if not getattr(
                        self, "_aasf_config_printed", False
                    ):
                        print(
                            "[AASF config] "
                            f"mode={action_smooth_mode} "
                            f"alpha={action_smooth_alpha:.6f} "
                            f"alpha_min={alpha_min:.6f} "
                            f"alpha_max={alpha_max:.6f} "
                            f"gain={gain:.6f} "
                            f"tau={threshold} "
                            f"tau_source={threshold_source}"
                        )
                        self._aasf_config_printed = True

                    debug_enabled = os.environ.get(
                        "DP_AASF_DEBUG", "0"
                    ) == "1"
                    debug_limit = int(os.environ.get(
                        "DP_AASF_DEBUG_LIMIT", "6"
                    ))
                    debug_count = int(getattr(
                        self, "_aasf_debug_chunks", 0
                    ))

                    if (
                        debug_enabled
                        and debug_count < debug_limit
                    ):
                        if raw_action.shape[1] > 1:
                            raw_step_mean = float(
                                np.linalg.norm(
                                    np.diff(
                                        raw_action,
                                        axis=1,
                                    ),
                                    axis=-1,
                                ).mean()
                            )
                            filtered_step_mean = float(
                                np.linalg.norm(
                                    np.diff(
                                        action,
                                        axis=1,
                                    ),
                                    axis=-1,
                                ).mean()
                            )
                        else:
                            raw_step_mean = 0.0
                            filtered_step_mean = 0.0

                        max_change = float(
                            np.linalg.norm(
                                action - raw_action,
                                axis=-1,
                            ).max()
                        )

                        trigger_fraction = (
                            trigger_count
                            / transition_count
                            if transition_count > 0
                            else 0.0
                        )

                        print(
                            "[AASF debug] "
                            f"chunk={debug_count} "
                            f"shape={raw_action.shape} "
                            f"raw_step_mean="
                            f"{raw_step_mean:.10f} "
                            f"filtered_step_mean="
                            f"{filtered_step_mean:.10f} "
                            f"trigger_fraction="
                            f"{trigger_fraction:.6f} "
                            f"max_change={max_change:.10f}"
                        )
                        self._aasf_debug_chunks = debug_count + 1

                # END SCALE-AWARE AASF (BlockPush)

                # BEGIN AASF FORENSIC TRACE: pre-step values
                if trace_this_chunk:
                    trace_raw_action = (
                        raw_action.copy()
                        if action_smooth_mode != "off"
                        else action.copy()
                    )
                    trace_executed_action = action.copy()

                    trace_alpha = np.full(
                        trace_raw_action.shape[:2] + (1,),
                        np.nan,
                        dtype=np.float32,
                    )
                    trace_trigger = np.zeros(
                        trace_raw_action.shape[:2] + (1,),
                        dtype=np.bool_,
                    )

                    if action_smooth_mode == "fixed":
                        trace_alpha[:, 1:, :] = float(
                            action_smooth_alpha
                        )

                    elif action_smooth_mode == "adaptive":
                        for trace_action_index in range(
                            1,
                            trace_raw_action.shape[1],
                        ):
                            trace_delta = np.linalg.norm(
                                trace_raw_action[
                                    :,
                                    trace_action_index,
                                ]
                                - trace_executed_action[
                                    :,
                                    trace_action_index - 1,
                                ],
                                axis=-1,
                                keepdims=True,
                            )

                            trace_mask = (
                                trace_delta > threshold
                            )
                            trace_excess = np.maximum(
                                trace_delta - threshold,
                                0.0,
                            ) / (threshold + 1e-12)

                            trace_adaptive_alpha = np.clip(
                                action_smooth_alpha
                                - gain * trace_excess,
                                alpha_min,
                                alpha_max,
                            )

                            trace_alpha[
                                :,
                                trace_action_index,
                                :,
                            ] = trace_adaptive_alpha
                            trace_trigger[
                                :,
                                trace_action_index,
                                :,
                            ] = trace_mask

                # END AASF FORENSIC TRACE: pre-step values

                # BEGIN VERIFIED BLOCKPUSH BSOP UPDATE
                if bsop_enabled:
                    if bsop_current_action_pred is None:
                        raise RuntimeError(
                            "BSOP current prediction state "
                            "was not initialized."
                        )

                    bsop_previous_action_pred = (
                        bsop_current_action_pred.copy()
                    )

                    bsop_previous_action_slice_start = int(
                        bsop_current_action_slice_start
                    )

                    bsop_previous_executed_chunk_length = int(
                        bsop_current_executed_chunk_length
                    )

                    bsop_previous_executed_last = (
                        action[:, -1].copy()
                    )
                # END VERIFIED BLOCKPUSH BSOP UPDATE

                # step env
                obs, reward, done, info = env.step(action)

                # BEGIN BLOCKPUSH ACTUAL EXECUTION ACCOUNTING
                if trace_this_chunk:
                    reward_histories_after_step = env.call(
                        "get_attr",
                        "reward",
                    )

                    current_executed_steps = np.asarray(
                        [
                            len(history)
                            for history
                            in reward_histories_after_step
                        ],
                        dtype=np.int64,
                    )

                    if current_executed_steps.shape != (
                        n_envs,
                    ):
                        raise RuntimeError(
                            "Unexpected BlockPush execution "
                            "counter shape: "
                            f"{current_executed_steps.shape}, "
                            f"expected {(n_envs,)}."
                        )

                    executed_steps_this_query = (
                        current_executed_steps
                        - trace_cumulative_executed_steps
                    )

                    planned_steps_this_query = int(
                        action.shape[1]
                    )

                    if np.any(
                        executed_steps_this_query < 0
                    ):
                        raise RuntimeError(
                            "BlockPush cumulative executed "
                            "step count decreased."
                        )

                    if np.any(
                        executed_steps_this_query
                        > planned_steps_this_query
                    ):
                        raise RuntimeError(
                            "BlockPush executed more actions "
                            "than were supplied in the chunk: "
                            f"executed="
                            f"{executed_steps_this_query}, "
                            f"planned="
                            f"{planned_steps_this_query}."
                        )

                    for local_index in range(
                        this_n_active_envs
                    ):
                        execution_count = int(
                            executed_steps_this_query[
                                local_index
                            ]
                        )

                        trace_planned_steps_per_query[
                            local_index
                        ].append(
                            planned_steps_this_query
                        )

                        trace_executed_steps_per_query[
                            local_index
                        ].append(
                            execution_count
                        )

                        if execution_count > 0:
                            trace_actual_executed_action_parts[
                                local_index
                            ].append(
                                action[
                                    local_index,
                                    :execution_count,
                                ].copy()
                            )

                    trace_cumulative_executed_steps = (
                        current_executed_steps
                    )
                # END BLOCKPUSH ACTUAL EXECUTION ACCOUNTING

                # BEGIN AASF FORENSIC TRACE: post-step values
                if trace_this_chunk:
                    trace_reward_array = np.asarray(reward)
                    trace_done_array = np.asarray(done)

                    for local_index in range(
                        this_n_active_envs
                    ):
                        seed = trace_chunk_seeds[local_index]

                        if (
                            trace_target_seeds
                            and seed not in trace_target_seeds
                        ):
                            continue

                        local_info = {
                            str(key): np.asarray(value).copy()
                            for key, value
                            in info[local_index].items()
                        }

                        trace_records[local_index].append({
                            "policy_call_index": len(
                                trace_records[local_index]
                            ),
                            "environment_step_start": (
                                trace_action_offset
                            ),
                            "planned_steps_this_query": (
                                np.int64(
                                    trace_planned_steps_per_query[
                                        local_index
                                    ][-1]
                                )
                            ),
                            "executed_steps_this_query": (
                                np.int64(
                                    trace_executed_steps_per_query[
                                        local_index
                                    ][-1]
                                )
                            ),

                            # Read-only BSOP mechanism
                            # diagnostics.
                            "bsop_boundary_available": (
                                np.bool_(
                                    bsop_diag_available[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_boundary_executed": (
                                np.bool_(
                                    bsop_diag_available[
                                        local_index
                                    ]
                                    and int(
                                        trace_executed_steps_per_query[
                                            local_index
                                        ][-1]
                                    ) > 0
                                )
                            ),
                            "bsop_raw_boundary_jump": (
                                np.float64(
                                    bsop_diag_raw_boundary_jump[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_projected_boundary_jump": (
                                np.float64(
                                    bsop_diag_projected_boundary_jump[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_correction": (
                                np.float64(
                                    bsop_diag_correction[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_relative_correction": (
                                np.float64(
                                    bsop_diag_relative_correction[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_lambda_star": (
                                np.float64(
                                    bsop_diag_lambda_star[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_lambda": (
                                np.float64(
                                    bsop_diag_lambda[
                                        local_index
                                    ]
                                )
                            ),
                            "bsop_active": (
                                np.bool_(
                                    bsop_diag_active[
                                        local_index
                                    ]
                                )
                            ),

                            "obs_before": (
                                trace_obs_before[
                                    local_index
                                ].copy()
                            ),
                            "raw_action": (
                                trace_raw_action[
                                    local_index
                                ].copy()
                            ),
                            "executed_action": (
                                trace_executed_action[
                                    local_index
                                ].copy()
                            ),
                            "adaptive_alpha": (
                                trace_alpha[
                                    local_index
                                ].copy()
                            ),
                            "trigger_mask": (
                                trace_trigger[
                                    local_index
                                ].copy()
                            ),
                            "obs_after": (
                                np.asarray(
                                    obs[local_index]
                                ).copy()
                            ),
                            "reward": (
                                np.asarray(
                                    trace_reward_array[
                                        local_index
                                    ]
                                ).copy()
                            ),
                            "done": (
                                np.asarray(
                                    trace_done_array[
                                        local_index
                                    ]
                                ).copy()
                            ),
                            "info": local_info,
                            "tau": (
                                float(threshold)
                                if action_smooth_mode
                                == "adaptive"
                                else np.nan
                            ),
                            "alpha_base": (
                                float(action_smooth_alpha)
                                if action_smooth_mode
                                != "off"
                                else np.nan
                            ),
                            "alpha_min": (
                                float(alpha_min)
                                if action_smooth_mode
                                != "off"
                                else np.nan
                            ),
                            "alpha_max": (
                                float(alpha_max)
                                if action_smooth_mode
                                != "off"
                                else np.nan
                            ),
                            "gain": (
                                float(gain)
                                if action_smooth_mode
                                != "off"
                                else np.nan
                            ),
                        })

                    trace_action_offset += int(
                        action.shape[1]
                    )
                # END AASF FORENSIC TRACE: post-step values

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # BEGIN AASF FORENSIC TRACE: save files
            if trace_this_chunk:
                trace_dir = os.environ.get(
                    "DP_AASF_TRACE_DIR",
                    os.path.join(
                        getattr(self, "output_dir", "."),
                        "aasf_trace",
                    ),
                )
                os.makedirs(trace_dir, exist_ok=True)

                for local_index in range(
                    this_n_active_envs
                ):
                    seed = trace_chunk_seeds[local_index]

                    if (
                        trace_target_seeds
                        and seed not in trace_target_seeds
                    ):
                        continue

                    records = trace_records[local_index]

                    if not records:
                        continue

                    def trace_stack(key):
                        values = [
                            np.asarray(record[key])
                            for record in records
                        ]
                        try:
                            return np.stack(values, axis=0)
                        except ValueError:
                            return np.asarray(
                                values,
                                dtype=object,
                            )

                    trace_path = os.path.join(
                        trace_dir,
                        (
                            f"seed_{seed}_"
                            f"{action_smooth_mode}.npz"
                        ),
                    )

                    planned_steps_per_query = np.asarray(
                        trace_planned_steps_per_query[
                            local_index
                        ],
                        dtype=np.int64,
                    )

                    executed_steps_per_query = np.asarray(
                        trace_executed_steps_per_query[
                            local_index
                        ],
                        dtype=np.int64,
                    )

                    actual_parts = (
                        trace_actual_executed_action_parts[
                            local_index
                        ]
                    )

                    if actual_parts:
                        actual_executed_action = (
                            np.concatenate(
                                actual_parts,
                                axis=0,
                            )
                        )
                    else:
                        example_action = np.asarray(
                            records[0][
                                "executed_action"
                            ]
                        )

                        actual_executed_action = np.empty(
                            (
                                0,
                                example_action.shape[-1],
                            ),
                            dtype=example_action.dtype,
                        )

                    actual_executed_steps = int(
                        np.sum(
                            executed_steps_per_query
                        )
                    )

                    total_planned_steps = int(
                        np.sum(
                            planned_steps_per_query
                        )
                    )

                    unexecuted_action_steps_total = (
                        total_planned_steps
                        - actual_executed_steps
                    )

                    if (
                        len(planned_steps_per_query)
                        != len(records)
                    ):
                        raise RuntimeError(
                            "Planned-step record count "
                            "does not match policy calls."
                        )

                    if (
                        len(executed_steps_per_query)
                        != len(records)
                    ):
                        raise RuntimeError(
                            "Executed-step record count "
                            "does not match policy calls."
                        )

                    if (
                        actual_executed_action.shape[0]
                        != actual_executed_steps
                    ):
                        raise RuntimeError(
                            "Actual action trajectory length "
                            "does not match executed count: "
                            f"trajectory="
                            f"{actual_executed_action.shape[0]}, "
                            f"count="
                            f"{actual_executed_steps}."
                        )

                    if unexecuted_action_steps_total < 0:
                        raise RuntimeError(
                            "Negative number of unexecuted "
                            "planned actions."
                        )

                    np.savez_compressed(
                        trace_path,
                        seed=np.asarray(
                            seed,
                            dtype=np.int64,
                        ),
                        mode=np.asarray(
                            action_smooth_mode
                        ),
                        execution_trace_version=np.asarray(
                            "actual_execution_trace_v1"
                        ),
                        planned_steps_per_query=(
                            planned_steps_per_query
                        ),
                        executed_steps_per_query=(
                            executed_steps_per_query
                        ),
                        actual_executed_action=(
                            actual_executed_action
                        ),
                        actual_executed_steps=np.asarray(
                            actual_executed_steps,
                            dtype=np.int64,
                        ),
                        unexecuted_action_steps_total=np.asarray(
                            unexecuted_action_steps_total,
                            dtype=np.int64,
                        ),

                        # BSOP configuration and mechanism
                        # records.
                        bsop_enabled=np.asarray(
                            bsop_enabled,
                            dtype=np.bool_,
                        ),
                        bsop_eta=np.asarray(
                            bsop_eta,
                            dtype=np.float64,
                        ),
                        bsop_lambda_max=np.asarray(
                            bsop_lambda_max,
                            dtype=np.float64,
                        ),
                        bsop_rho=np.asarray(
                            bsop_rho,
                            dtype=np.float64,
                        ),
                        bsop_kappa=np.asarray(
                            bsop_kappa,
                            dtype=np.float64,
                        ),
                        bsop_boundary_available=trace_stack(
                            "bsop_boundary_available"
                        ),
                        bsop_boundary_executed=trace_stack(
                            "bsop_boundary_executed"
                        ),
                        bsop_raw_boundary_jump=trace_stack(
                            "bsop_raw_boundary_jump"
                        ),
                        bsop_projected_boundary_jump=trace_stack(
                            "bsop_projected_boundary_jump"
                        ),
                        bsop_correction=trace_stack(
                            "bsop_correction"
                        ),
                        bsop_relative_correction=trace_stack(
                            "bsop_relative_correction"
                        ),
                        bsop_lambda_star=trace_stack(
                            "bsop_lambda_star"
                        ),
                        bsop_lambda=trace_stack(
                            "bsop_lambda"
                        ),
                        bsop_active=trace_stack(
                            "bsop_active"
                        ),

                        policy_call_index=np.asarray(
                            [
                                record["policy_call_index"]
                                for record in records
                            ],
                            dtype=np.int64,
                        ),
                        environment_step_start=np.asarray(
                            [
                                record[
                                    "environment_step_start"
                                ]
                                for record in records
                            ],
                            dtype=np.int64,
                        ),
                        obs_before=trace_stack(
                            "obs_before"
                        ),
                        raw_action=trace_stack(
                            "raw_action"
                        ),
                        executed_action=trace_stack(
                            "executed_action"
                        ),
                        adaptive_alpha=trace_stack(
                            "adaptive_alpha"
                        ),
                        trigger_mask=trace_stack(
                            "trigger_mask"
                        ),
                        obs_after=trace_stack(
                            "obs_after"
                        ),
                        reward=trace_stack("reward"),
                        done=trace_stack("done"),
                        tau=np.asarray(
                            [
                                record["tau"]
                                for record in records
                            ],
                            dtype=np.float64,
                        ),
                        alpha_base=np.asarray(
                            [
                                record["alpha_base"]
                                for record in records
                            ],
                            dtype=np.float64,
                        ),
                        alpha_min=np.asarray(
                            [
                                record["alpha_min"]
                                for record in records
                            ],
                            dtype=np.float64,
                        ),
                        alpha_max=np.asarray(
                            [
                                record["alpha_max"]
                                for record in records
                            ],
                            dtype=np.float64,
                        ),
                        gain=np.asarray(
                            [
                                record["gain"]
                                for record in records
                            ],
                            dtype=np.float64,
                        ),
                        info=np.asarray(
                            [
                                record["info"]
                                for record in records
                            ],
                            dtype=object,
                        ),
                    )

                    print(
                        "[AASF trace] "
                        f"seed={seed} "
                        f"mode={action_smooth_mode} "
                        f"policy_calls={len(records)} "
                        f"path={trace_path}"
                    )
            # END AASF FORENSIC TRACE: save files

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # log
        total_rewards = collections.defaultdict(list)
        total_p1 = collections.defaultdict(list)
        total_p2 = collections.defaultdict(list)
        prefix_event_counts = collections.defaultdict(lambda :collections.defaultdict(lambda : 0))
        prefix_counts = collections.defaultdict(lambda : 0)

        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.unique(this_rewards).sum() # (0, 0.49, 0.51)
            p1 = total_reward > 0.4
            p2 = total_reward > 0.9

            total_rewards[prefix].append(total_reward)
            total_p1[prefix].append(p1)
            total_p2[prefix].append(p2)
            log_data[prefix+f'sim_max_reward_{seed}'] = total_reward

            # aggregate event counts
            prefix_counts[prefix] += 1
            for key, value in last_info[i].items():
                delta_count = 1 if value > 0 else 0
                prefix_event_counts[prefix][key] += delta_count

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in total_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p1.items():
            name = prefix+'p1'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p2.items():
            name = prefix+'p2'
            value = np.mean(value)
            log_data[name] = value

        # summarize probabilities
        for prefix, events in prefix_event_counts.items():
            prefix_count = prefix_counts[prefix]
            for event, count in events.items():
                prob = count / prefix_count
                key = prefix + event
                log_data[key] = prob

        return log_data
