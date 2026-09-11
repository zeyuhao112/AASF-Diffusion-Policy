import wandb
import numpy as np
import torch
import collections
import hashlib
import pathlib
import tqdm
import dill
import math
import os
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
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

        # Gym compatibility: image observations use a mapping observation space
        # that is incompatible with Gym shared-memory vectorization.
        env = AsyncVectorEnv(
            env_fns,
            shared_memory=False,
        )

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

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

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # BEGIN PUSHT IMAGE PAIRED POLICY SEED CONFIG
        #
        # Optional common-random-number protocol.
        # Disabled by default so this instrumentation has
        # zero effect on the original evaluation path.
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
        # END PUSHT IMAGE PAIRED POLICY SEED CONFIG

        # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE CONFIG
        #
        # Read-only forensic instrumentation. Disabled by
        # default and must never participate in control.
        trace_enabled = (
            os.environ.get(
                "DP_IMAGE_TRACE",
                "0",
            ) == "1"
        )

        trace_root = pathlib.Path(
            os.environ.get(
                "DP_IMAGE_TRACE_DIR",
                str(
                    pathlib.Path(self.output_dir)
                    / "image_execution_trace"
                ),
            )
        )

        trace_seed_spec = os.environ.get(
            "DP_IMAGE_TRACE_SEEDS",
            "",
        )

        trace_target_seeds = {
            int(item.strip())
            for item in trace_seed_spec.split(",")
            if item.strip()
        }

        trace_method = os.environ.get(
            "DP_IMAGE_TRACE_METHOD",
            "unspecified",
        ).strip()
        # END PUSHT IMAGE ACTUAL EXECUTION TRACE CONFIG

        # BEGIN PUSHT IMAGE BSOP CONFIG
        #
        # Modality transfer without retuning.
        # These are the frozen Push-T lowdim BSOP
        # confirmatory parameters.
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

        bsop_rho = float(
            os.environ.get(
                "DP_BSOP_RHO",
                "5.0",
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
        # END PUSHT IMAGE BSOP CONFIG

        # plan for rollout
        n_envs = len(self.env_fns)

        if paired_policy_seed_enabled and n_envs != 1:
            raise RuntimeError(
                "DP_PAIRED_POLICY_SEED requires "
                "DP_N_ENVS=1."
            )

        if trace_enabled and n_envs != 1:
            raise RuntimeError(
                "DP_IMAGE_TRACE requires DP_N_ENVS=1."
            )

        if trace_enabled:
            trace_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        if bsop_enabled and n_envs != 1:
            raise RuntimeError(
                "PushT Image BSOP requires "
                "DP_N_ENVS=1."
            )

        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

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

            # BEGIN PUSHT IMAGE PAIRED POLICY SEED
            #
            # With n_envs=1 each chunk is one episode.
            # Original and modified methods therefore receive
            # the same diffusion-noise stream for the same
            # environment seed.
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
            # END PUSHT IMAGE PAIRED POLICY SEED

            # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE
            trace_episode_seed = int(
                self.env_seeds[start]
            )

            trace_this_episode = (
                trace_enabled
                and (
                    not trace_target_seeds
                    or trace_episode_seed
                    in trace_target_seeds
                )
            )

            trace_records = []
            trace_actual_action_parts = []
            trace_cumulative_executed_steps = 0
            # END PUSHT IMAGE ACTUAL EXECUTION TRACE

            # BEGIN PUSHT IMAGE BSOP STATE
            bsop_previous_action_pred = None
            bsop_previous_action_slice_start = None
            bsop_previous_executed_chunk_length = None
            bsop_previous_executed_last = None
            # END PUSHT IMAGE BSOP STATE

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE
                if trace_this_episode:
                    trace_hasher = hashlib.sha256()

                    for trace_key in sorted(
                        np_obs_dict.keys()
                    ):
                        trace_array = np.ascontiguousarray(
                            np_obs_dict[trace_key]
                        )

                        trace_hasher.update(
                            str(trace_key).encode("utf-8")
                        )
                        trace_hasher.update(
                            str(trace_array.dtype).encode(
                                "utf-8"
                            )
                        )
                        trace_hasher.update(
                            str(trace_array.shape).encode(
                                "utf-8"
                            )
                        )
                        trace_hasher.update(
                            trace_array.tobytes()
                        )

                    trace_obs_sha256 = (
                        trace_hasher.hexdigest()
                    )

                    if "agent_pos" in np_obs_dict:
                        trace_agent_pos_before = (
                            np.asarray(
                                np_obs_dict["agent_pos"][0]
                            ).copy()
                        )
                    else:
                        trace_agent_pos_before = (
                            np.empty(
                                (0,),
                                dtype=np.float32,
                            )
                        )
                # END PUSHT IMAGE ACTUAL EXECUTION TRACE

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

                # Original diffusion-policy command before
                # any inference-time post-processing.
                raw_action = action.copy()

                # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE
                if trace_this_episode:
                    trace_raw_action = (
                        raw_action.copy()
                    )

                    if "action_pred" in np_action_dict:
                        trace_action_pred = (
                            np.asarray(
                                np_action_dict[
                                    "action_pred"
                                ]
                            ).copy()
                        )
                    else:
                        trace_action_pred = np.empty(
                            (
                                action.shape[0],
                                0,
                                action.shape[-1],
                            ),
                            dtype=action.dtype,
                        )
                # END PUSHT IMAGE ACTUAL EXECUTION TRACE

                # BEGIN PUSHT IMAGE BSOP DIAGNOSTICS
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
                # END PUSHT IMAGE BSOP DIAGNOSTICS

                # BEGIN PUSHT IMAGE BSOP CORE
                bsop_current_action_pred = None
                bsop_current_action_slice_start = None
                bsop_current_executed_chunk_length = None

                if bsop_enabled:
                    if "action_pred" not in np_action_dict:
                        raise RuntimeError(
                            "Image BSOP requires policy "
                            "output 'action_pred'."
                        )

                    bsop_current_action_pred = np.asarray(
                        np_action_dict["action_pred"]
                    )

                    if (
                        bsop_current_action_pred.ndim
                        != 3
                    ):
                        raise RuntimeError(
                            "Image BSOP expected "
                            "action_pred shape "
                            "(batch, horizon, action_dim), "
                            f"got "
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

                    # DiffusionUnetHybridImagePolicy uses:
                    #
                    # start = To - 1
                    #
                    # for the full-horizon prediction.
                    if bsop_pred_steps_only:
                        bsop_policy_slice_start = 0
                    else:
                        bsop_policy_slice_start = (
                            bsop_n_obs_steps - 1
                        )

                    bsop_current_action_slice_start = (
                        bsop_policy_slice_start
                        + int(
                            getattr(
                                self,
                                "n_latency_steps",
                                0,
                            )
                        )
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
                            "Image BSOP action_pred slice "
                            "has wrong shape: "
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
                            "Image BSOP action_pred "
                            "alignment check failed. "
                            f"max_abs_diff="
                            f"{bsop_max_diff:.9g}"
                        )

                    if (
                        bsop_previous_action_pred
                        is not None
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
                                "Image BSOP previous overlap "
                                "index is outside action_pred: "
                                f"index="
                                f"{bsop_previous_overlap_index}, "
                                f"horizon="
                                f"{bsop_previous_action_pred.shape[1]}."
                            )

                        # Float64 projection geometry.
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
                                "Image BSOP vector shapes "
                                "differ: "
                                f"y={bsop_y.shape}, "
                                f"a={bsop_a.shape}, "
                                f"p={bsop_p.shape}."
                            )

                        bsop_d = (
                            bsop_a - bsop_y
                        )
                        bsop_q = (
                            bsop_p - bsop_a
                        )

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

                        bsop_lambda_star[
                            bsop_valid
                        ] = (
                            -np.sum(
                                bsop_d[bsop_valid]
                                * bsop_q[bsop_valid],
                                axis=-1,
                            )
                            / bsop_q_norm_sq[
                                bsop_valid
                            ]
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

                        # Absolute correction radius:
                        #
                        # ||b-a|| <= rho
                        bsop_lambda = np.minimum(
                            bsop_lambda,
                            bsop_rho
                            / (
                                bsop_q_norm
                                + 1.0e-12
                            ),
                        )

                        # Relative-response protection:
                        #
                        # ||b-a||
                        # <= kappa * ||a-y||
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

                        bsop_projected_jump = (
                            np.linalg.norm(
                                bsop_projected_first
                                - bsop_y,
                                axis=-1,
                            )
                        )

                        bsop_correction = (
                            np.linalg.norm(
                                bsop_projected_first
                                - bsop_a,
                                axis=-1,
                            )
                        )

                        # Geometric guarantees before cast.
                        if np.any(
                            bsop_projected_jump
                            > bsop_baseline_jump
                            + 1.0e-8
                        ):
                            raise RuntimeError(
                                "Image BSOP boundary "
                                "guarantee failed before "
                                "dtype cast."
                            )

                        if np.any(
                            bsop_correction
                            > bsop_rho
                            + 1.0e-8
                        ):
                            raise RuntimeError(
                                "Image BSOP absolute "
                                "correction constraint "
                                "failed."
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
                                "Image BSOP relative "
                                "response constraint "
                                "failed."
                            )

                        action = action.copy()

                        action[:, 0] = (
                            bsop_projected_first.astype(
                                action.dtype,
                                copy=False,
                            )
                        )

                        # Recheck the actual float32 command.
                        bsop_cast_jump = np.linalg.norm(
                            np.asarray(
                                action[:, 0],
                                dtype=np.float64,
                            )
                            - bsop_y,
                            axis=-1,
                        )

                        bsop_cast_correction = (
                            np.linalg.norm(
                                np.asarray(
                                    action[:, 0],
                                    dtype=np.float64,
                                )
                                - bsop_a,
                                axis=-1,
                            )
                        )

                        if np.any(
                            bsop_cast_jump
                            > bsop_baseline_jump
                            + 1.0e-4
                        ):
                            raise RuntimeError(
                                "Image BSOP boundary "
                                "guarantee failed after "
                                "dtype cast."
                            )

                        if np.any(
                            bsop_cast_correction
                            > bsop_rho
                            + 1.0e-4
                        ):
                            raise RuntimeError(
                                "Image BSOP correction "
                                "radius failed after "
                                "dtype cast."
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
                                "Image BSOP relative "
                                "response failed after "
                                "dtype cast."
                            )

                        # Read-only mechanism diagnostics.
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
                # END PUSHT IMAGE BSOP CORE

                # BEGIN PUSHT IMAGE AASF CORE
                # Inference-time action smoothing applied only to the
                # action chunk actually submitted to MultiStepWrapper.
                # This mirrors the frozen Push-T low-dimensional AASF
                # implementation. The first action is preserved exactly
                # and filter state is reset for every policy query.
                action_smooth_mode = os.environ.get(
                    "DP_ACTION_SMOOTH_MODE",
                    "fixed",
                )
                action_smooth_alpha = float(
                    os.environ.get(
                        "DP_ACTION_SMOOTH_ALPHA",
                        "1.0",
                    )
                )

                action_smooth_enabled = (
                    (action_smooth_alpha < 1.0)
                    or (action_smooth_mode == "sma")
                )

                # BSOP and AASF are separate experimental methods.
                # Never allow an accidental combined intervention.
                if bsop_enabled and action_smooth_enabled:
                    raise RuntimeError(
                        "BSOP and action smoothing cannot be "
                        "enabled simultaneously."
                    )

                if action_smooth_enabled:
                    aasf_input_action = action.copy()
                    smoothed_action = action.copy()

                    if action_smooth_mode == "sma":
                        sma_window = int(
                            os.environ.get(
                                "DP_ACTION_SMA_WINDOW",
                                "3",
                            )
                        )
                        sma_window = max(1, sma_window)

                        for t in range(
                            smoothed_action.shape[1]
                        ):
                            start = max(
                                0,
                                t - sma_window + 1,
                            )
                            smoothed_action[:, t] = (
                                action[:, start:t+1]
                                .mean(axis=1)
                            )

                    elif action_smooth_mode == "adaptive":
                        alpha_min = float(
                            os.environ.get(
                                "DP_ACTION_SMOOTH_ALPHA_MIN",
                                "0.45",
                            )
                        )
                        alpha_max = float(
                            os.environ.get(
                                "DP_ACTION_SMOOTH_ALPHA_MAX",
                                str(action_smooth_alpha),
                            )
                        )
                        threshold = float(
                            os.environ.get(
                                "DP_ACTION_SMOOTH_THRESHOLD",
                                "10.0",
                            )
                        )
                        gain = float(
                            os.environ.get(
                                "DP_ACTION_SMOOTH_GAIN",
                                "0.25",
                            )
                        )

                        if not (
                            0.0
                            <= alpha_min
                            <= alpha_max
                            <= 1.0
                        ):
                            raise RuntimeError(
                                "Invalid AASF alpha bounds: "
                                f"alpha_min={alpha_min}, "
                                f"alpha_max={alpha_max}."
                            )

                        if threshold <= 0.0:
                            raise RuntimeError(
                                "DP_ACTION_SMOOTH_THRESHOLD "
                                "must be positive."
                            )

                        if gain < 0.0:
                            raise RuntimeError(
                                "DP_ACTION_SMOOTH_GAIN "
                                "must be non-negative."
                            )

                        for t in range(
                            1,
                            smoothed_action.shape[1],
                        ):
                            delta = np.linalg.norm(
                                action[:, t]
                                - smoothed_action[:, t-1],
                                axis=-1,
                                keepdims=True,
                            )

                            excess = np.maximum(
                                delta - threshold,
                                0.0,
                            ) / (
                                threshold + 1e-6
                            )

                            adaptive_alpha = (
                                action_smooth_alpha
                                - gain * excess
                            )

                            adaptive_alpha = np.clip(
                                adaptive_alpha,
                                alpha_min,
                                alpha_max,
                            )

                            smoothed_action[:, t] = (
                                adaptive_alpha
                                * action[:, t]
                                + (
                                    1.0
                                    - adaptive_alpha
                                )
                                * smoothed_action[:, t-1]
                            )

                    else:
                        # Fixed EMA branch. This intentionally mirrors
                        # the frozen low-dimensional implementation.
                        for t in range(
                            1,
                            smoothed_action.shape[1],
                        ):
                            smoothed_action[:, t] = (
                                action_smooth_alpha
                                * action[:, t]
                                + (
                                    1.0
                                    - action_smooth_alpha
                                )
                                * smoothed_action[:, t-1]
                            )

                    # AASF must preserve the first action exactly.
                    if not np.array_equal(
                        smoothed_action[:, 0],
                        aasf_input_action[:, 0],
                    ):
                        raise RuntimeError(
                            "AASF first-action preservation "
                            "check failed."
                        )

                    action = smoothed_action
                # END PUSHT IMAGE AASF CORE

                # Complete chunk actually submitted to
                # MultiStepWrapper after BSOP.
                if trace_this_episode:
                    trace_executed_chunk = (
                        np.asarray(action).copy()
                    )

                # step env
                obs, reward, done, info = env.step(action)

                # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE
                if trace_this_episode:
                    trace_reward_histories = env.call(
                        "get_attr",
                        "reward",
                    )

                    trace_current_executed_steps = int(
                        len(
                            trace_reward_histories[0]
                        )
                    )

                    trace_executed_this_query = (
                        trace_current_executed_steps
                        - trace_cumulative_executed_steps
                    )

                    trace_planned_this_query = int(
                        action.shape[1]
                    )

                    if trace_executed_this_query < 0:
                        raise RuntimeError(
                            "Actual executed-action count "
                            "decreased."
                        )

                    if (
                        trace_executed_this_query
                        > trace_planned_this_query
                    ):
                        raise RuntimeError(
                            "Environment executed more "
                            "actions than were supplied."
                        )

                    if trace_executed_this_query > 0:
                        trace_actual_action_parts.append(
                            trace_executed_chunk[
                                0,
                                :trace_executed_this_query,
                            ].copy()
                        )

                    trace_records.append({
                        "policy_call_index": np.int64(
                            len(trace_records)
                        ),
                        "environment_step_start": np.int64(
                            trace_cumulative_executed_steps
                        ),
                        "planned_steps_this_query": np.int64(
                            trace_planned_this_query
                        ),
                        "executed_steps_this_query": np.int64(
                            trace_executed_this_query
                        ),
                        # BSOP mechanism diagnostics.
                        "bsop_boundary_available": (
                            np.bool_(
                                bsop_diag_available[0]
                            )
                        ),
                        "bsop_boundary_executed": (
                            np.bool_(
                                bsop_diag_available[0]
                                and (
                                    trace_executed_this_query
                                    > 0
                                )
                            )
                        ),
                        "bsop_raw_boundary_jump": (
                            np.float64(
                                bsop_diag_raw_boundary_jump[
                                    0
                                ]
                            )
                        ),
                        "bsop_projected_boundary_jump": (
                            np.float64(
                                bsop_diag_projected_boundary_jump[
                                    0
                                ]
                            )
                        ),
                        "bsop_correction": (
                            np.float64(
                                bsop_diag_correction[0]
                            )
                        ),
                        "bsop_relative_correction": (
                            np.float64(
                                bsop_diag_relative_correction[
                                    0
                                ]
                            )
                        ),
                        "bsop_lambda_star": (
                            np.float64(
                                bsop_diag_lambda_star[0]
                            )
                        ),
                        "bsop_lambda": (
                            np.float64(
                                bsop_diag_lambda[0]
                            )
                        ),
                        "bsop_active": (
                            np.bool_(
                                bsop_diag_active[0]
                            )
                        ),

                        "obs_sha256": (
                            trace_obs_sha256
                        ),
                        "agent_pos_before": (
                            trace_agent_pos_before.copy()
                        ),
                        "raw_action": (
                            trace_raw_action[0].copy()
                        ),
                        "action_pred": (
                            trace_action_pred[0].copy()
                        ),
                        "executed_action": (
                            trace_executed_chunk[0].copy()
                        ),
                        "reward": np.asarray(
                            reward
                        ).reshape(-1)[0].copy(),
                        "done": np.asarray(
                            done
                        ).reshape(-1)[0].copy(),
                    })

                    trace_cumulative_executed_steps = (
                        trace_current_executed_steps
                    )
                # END PUSHT IMAGE ACTUAL EXECUTION TRACE

                done = np.all(done)

                # BEGIN PUSHT IMAGE BSOP STATE UPDATE
                if bsop_enabled and not done:
                    if (
                        bsop_current_action_pred
                        is None
                    ):
                        raise RuntimeError(
                            "Image BSOP current "
                            "prediction state was "
                            "not initialized."
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
                # END PUSHT IMAGE BSOP STATE UPDATE

                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # BEGIN PUSHT IMAGE ACTUAL EXECUTION TRACE SAVE
            if trace_this_episode:
                if not trace_records:
                    raise RuntimeError(
                        "Image execution trace recorded "
                        "no policy calls."
                    )

                planned_steps_per_query = np.asarray(
                    [
                        x["planned_steps_this_query"]
                        for x in trace_records
                    ],
                    dtype=np.int64,
                )

                executed_steps_per_query = np.asarray(
                    [
                        x["executed_steps_this_query"]
                        for x in trace_records
                    ],
                    dtype=np.int64,
                )

                if trace_actual_action_parts:
                    actual_executed_action = (
                        np.concatenate(
                            trace_actual_action_parts,
                            axis=0,
                        )
                    )
                else:
                    example_action = np.asarray(
                        trace_records[0][
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

                unexecuted_action_steps_total = int(
                    total_planned_steps
                    - actual_executed_steps
                )

                if (
                    actual_executed_action.shape[0]
                    != actual_executed_steps
                ):
                    raise RuntimeError(
                        "Actual executed trajectory "
                        "length/count mismatch."
                    )

                if (
                    unexecuted_action_steps_total
                    < 0
                ):
                    raise RuntimeError(
                        "Negative unexecuted action "
                        "count."
                    )

                def trace_stack(key):
                    return np.stack(
                        [
                            np.asarray(x[key])
                            for x in trace_records
                        ],
                        axis=0,
                    )

                trace_path = (
                    trace_root
                    / (
                        f"seed_{trace_episode_seed}_"
                        f"{trace_method}.npz"
                    )
                )

                np.savez_compressed(
                    trace_path,
                    seed=np.asarray(
                        trace_episode_seed,
                        dtype=np.int64,
                    ),
                    method=np.asarray(
                        trace_method
                    ),
                    execution_trace_version=np.asarray(
                        "pusht_image_actual_execution_v1"
                    ),
                    paired_policy_seed_enabled=np.asarray(
                        paired_policy_seed_enabled,
                        dtype=np.bool_,
                    ),
                    episode_policy_seed=np.asarray(
                        (
                            episode_policy_seed
                            if episode_policy_seed
                            is not None
                            else -1
                        ),
                        dtype=np.int64,
                    ),
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

                    policy_call_index=trace_stack(
                        "policy_call_index"
                    ),
                    environment_step_start=trace_stack(
                        "environment_step_start"
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
                    unexecuted_action_steps_total=(
                        np.asarray(
                            unexecuted_action_steps_total,
                            dtype=np.int64,
                        )
                    ),
                    obs_sha256=np.asarray(
                        [
                            x["obs_sha256"]
                            for x in trace_records
                        ]
                    ),
                    agent_pos_before=trace_stack(
                        "agent_pos_before"
                    ),
                    raw_action=trace_stack(
                        "raw_action"
                    ),
                    action_pred=trace_stack(
                        "action_pred"
                    ),
                    executed_action=trace_stack(
                        "executed_action"
                    ),
                    reward=trace_stack(
                        "reward"
                    ),
                    done=trace_stack(
                        "done"
                    ),
                )

                print(
                    "[Image execution trace] "
                    f"seed={trace_episode_seed} "
                    f"method={trace_method} "
                    f"policy_calls={len(trace_records)} "
                    f"actual_steps="
                    f"{actual_executed_steps} "
                    f"path={trace_path}"
                )
            # END PUSHT IMAGE ACTUAL EXECUTION TRACE SAVE

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
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
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
