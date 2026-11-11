import wandb
import numpy as np
from scipy.signal import savgol_filter
from diffusion_policy.common.temporal_ensemble import (
    temporal_ensemble_current_chunk,
    append_temporal_ensemble_record,
)
import torch
import collections
import os
import time
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
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
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
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
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
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
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_action_trajs = [None] * n_inits

        # Optional paper trace instrumentation.
        trace_enabled = (
            os.environ.get("DP_TRACE_ENABLE", "0") == "1"
        )
        trace_root = pathlib.Path(
            os.environ.get(
                "DP_TRACE_DIR",
                "paper_artifacts/traces/default",
            )
        )
        trace_method = os.environ.get(
            "DP_TRACE_METHOD",
            "unspecified",
        )
        trace_warmup_queries = int(
            os.environ.get(
                "DP_TRACE_WARMUP_QUERIES",
                "3",
            )
        )

        # Read-only cross-query prediction diagnostic.
        # This diagnostic must never modify policy outputs
        # or actions sent to the environment.
        overlap_diag_enabled = (
            os.environ.get(
                "DP_OVERLAP_DIAG_ENABLE",
                "0",
            ) == "1"
        )

        if overlap_diag_enabled and not trace_enabled:
            raise RuntimeError(
                "DP_OVERLAP_DIAG_ENABLE=1 requires "
                "DP_TRACE_ENABLE=1."
            )

        # Boundary-Safe Overlap Projection (BSOP).
        #
        # The first action of a newly predicted chunk may be
        # projected toward the previous query's prediction
        # for the same physical time. Projection is accepted
        # only along the segment that cannot increase the
        # current chunk-boundary action jump.
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
                "10.0",
            )
        )
        bsop_kappa = float(
            os.environ.get(
                "DP_BSOP_KAPPA",
                "1.0",
            )
        )

        if not (0.0 <= bsop_eta <= 1.0):
            raise ValueError(
                "DP_BSOP_ETA must lie in [0, 1]."
            )

        if not (0.0 <= bsop_lambda_max <= 1.0):
            raise ValueError(
                "DP_BSOP_LAMBDA_MAX must lie in [0, 1]."
            )

        if bsop_rho < 0.0:
            raise ValueError(
                "DP_BSOP_RHO must be non-negative."
            )

        if not (0.0 <= bsop_kappa <= 1.0):
            raise ValueError(
                "DP_BSOP_KAPPA must lie in [0, 1]."
            )

        # Temporal Ensembling strong baseline.
        #
        # Predictions from current and previous policy queries
        # are aligned by absolute actually-executed control-step
        # index. No future environment observation is accessed.
        te_enabled = (
            os.environ.get(
                "DP_TE_ENABLE",
                "0",
            ) == "1"
        )

        te_beta = float(
            os.environ.get(
                "DP_TE_BETA",
                "0.0",
            )
        )

        te_max_history_raw = int(
            os.environ.get(
                "DP_TE_MAX_HISTORY",
                "0",
            )
        )

        if te_beta < 0.0:
            raise ValueError(
                "DP_TE_BETA must be non-negative."
            )

        if te_max_history_raw < 0:
            raise ValueError(
                "DP_TE_MAX_HISTORY must be >= 0."
            )

        # Zero means that no explicit query-history cap is
        # imposed. Predictions outside their temporal horizon
        # naturally stop contributing.
        te_max_history = (
            None
            if te_max_history_raw == 0
            else te_max_history_raw
        )

        # A formal TE baseline must be isolated from BSOP and
        # the existing within-chunk smoothing baselines.
        te_action_smooth_mode = os.environ.get(
            "DP_ACTION_SMOOTH_MODE",
            "fixed",
        )
        te_action_smooth_alpha = float(
            os.environ.get(
                "DP_ACTION_SMOOTH_ALPHA",
                "1.0",
            )
        )

        te_other_smoother_active = (
            te_action_smooth_alpha < 1.0
            or te_action_smooth_mode in (
                "sma",
                "sg",
            )
        )

        sg_enabled = (
            te_action_smooth_mode == "sg"
        )

        if sg_enabled and bsop_enabled:
            raise RuntimeError(
                "Savitzky-Golay baseline must be "
                "evaluated without BSOP."
            )

        if te_enabled and bsop_enabled:
            raise RuntimeError(
                "Temporal Ensemble and BSOP must not be "
                "enabled simultaneously."
            )

        if (
            te_enabled
            and te_other_smoother_active
        ):
            raise RuntimeError(
                "Temporal Ensemble must be evaluated without "
                "EMA/AASF/SMA post-processing."
            )

        if trace_enabled:
            trace_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        all_raw_action_trajs = [None] * n_inits
        all_d_trajs = [None] * n_inits
        all_alpha_trajs = [None] * n_inits
        all_chunk_starts = [None] * n_inits
        all_policy_times_ms = [None] * n_inits
        all_postprocess_times_ms = [None] * n_inits
        all_decision_times_ms = [None] * n_inits

        # Strict actual-execution records. Legacy action
        # trajectories remain unchanged for reproducibility.
        all_planned_steps_per_query = [None] * n_inits
        all_executed_steps_per_query = [None] * n_inits
        all_actual_executed_action_trajs = [None] * n_inits

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

            # Optional common-random-number protocol.
            #
            # With n_envs=1, each chunk corresponds to exactly
            # one environment episode. Original and modified
            # methods receive the same PyTorch diffusion-noise
            # stream for the same environment seed.
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

            if paired_policy_seed_enabled:
                if n_envs != 1:
                    raise RuntimeError(
                        "DP_PAIRED_POLICY_SEED requires "
                        "DP_N_ENVS=1."
                    )

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

            action_history = []

            raw_action_history = []
            d_history = []
            alpha_history = []
            policy_time_history = []
            postprocess_time_history = []
            decision_time_history = []
            chunk_length_history = []

            # Actual control-action execution histories.
            #
            # MultiStepWrapper.reward receives exactly one
            # entry for every action passed to the wrapped
            # PushTEnv.step(). Its length is therefore the
            # cumulative number of truly executed actions.
            planned_steps_per_query_history = []
            executed_steps_per_query_history = []
            cumulative_executed_steps = np.zeros(
                (n_envs,),
                dtype=np.int64,
            )
            executed_action_parts = [
                []
                for _ in range(n_envs)
            ]

            # Temporal Ensemble state.
            #
            # exec_start values below always come from the
            # actual execution counter, never from the planned
            # chunk length.
            te_history = []
            te_exec_start_history = []
            te_executed_steps_history = []
            te_candidate_count_history = []
            te_source_age_max_history = []
            te_query_index_history = []

            # Zero-impact overlap diagnostic histories.
            overlap_action_pred_history = []
            overlap_slice_start_history = []
            overlap_horizon_history = []
            overlap_chunk_length_history = []

            overlap_previous_prediction_history = []
            overlap_current_prediction_history = []
            overlap_delta_l2_history = []
            overlap_length_history = []

            overlap_raw_boundary_jump_history = []
            overlap_filtered_boundary_jump_history = []

            previous_action_pred = None
            previous_raw_action = None
            previous_filtered_action = None
            previous_action_slice_start = None

            # BSOP state is independent of the optional
            # overlap diagnostic state.
            bsop_previous_action_pred = None
            bsop_previous_executed_last = None
            bsop_previous_action_slice_start = None
            bsop_previous_executed_chunk_length = None

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # Run policy and measure synchronized inference time.
                if trace_enabled:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    decision_start_ns = time.perf_counter_ns()
                    policy_start_ns = decision_start_ns

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                if trace_enabled:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    policy_end_ns = time.perf_counter_ns()

                # Device transfer.
                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to('cpu').numpy()
                )

                # Handle latency_steps by discarding the first steps.
                action = np_action_dict[
                    'action'
                ][:, self.n_latency_steps:]

                raw_action = action.copy()

                # ----------------------------------------------------
                # Temporal Ensembling baseline
                # ----------------------------------------------------
                te_exec_start_this_query = None

                if te_enabled:
                    if "action_pred" not in np_action_dict:
                        raise RuntimeError(
                            "Temporal Ensemble requires policy "
                            "output 'action_pred'."
                        )

                    te_current_action_pred = np.asarray(
                        np_action_dict["action_pred"]
                    )

                    if te_current_action_pred.ndim != 3:
                        raise RuntimeError(
                            "Temporal Ensemble expected "
                            "action_pred shape "
                            "(batch, horizon, action_dim), got "
                            f"{te_current_action_pred.shape}."
                        )

                    te_pred_steps_only = bool(
                        getattr(
                            policy,
                            "pred_action_steps_only",
                            False,
                        )
                    )

                    te_n_obs_steps = int(
                        getattr(
                            policy,
                            "n_obs_steps",
                            self.n_obs_steps,
                        )
                    )

                    te_oa_convention = bool(
                        getattr(
                            policy,
                            "oa_step_convention",
                            False,
                        )
                    )

                    if te_pred_steps_only:
                        te_policy_slice_start = 0
                    else:
                        te_policy_slice_start = (
                            te_n_obs_steps
                            - int(te_oa_convention)
                        )

                    te_action_slice_start = (
                        te_policy_slice_start
                        + int(self.n_latency_steps)
                    )

                    te_chunk_length = int(
                        raw_action.shape[1]
                    )

                    te_action_slice_end = (
                        te_action_slice_start
                        + te_chunk_length
                    )

                    te_expected_raw_action = (
                        te_current_action_pred[
                            :,
                            te_action_slice_start:
                            te_action_slice_end,
                        ]
                    )

                    if (
                        te_expected_raw_action.shape
                        != raw_action.shape
                    ):
                        raise RuntimeError(
                            "Temporal Ensemble inferred "
                            "action_pred slice has wrong shape: "
                            f"action_pred="
                            f"{te_current_action_pred.shape}, "
                            f"slice=["
                            f"{te_action_slice_start}:"
                            f"{te_action_slice_end}], "
                            f"expected="
                            f"{te_expected_raw_action.shape}, "
                            f"returned={raw_action.shape}."
                        )

                    if not np.array_equal(
                        te_expected_raw_action,
                        raw_action,
                    ):
                        te_alignment_max_diff = float(
                            np.max(
                                np.abs(
                                    te_expected_raw_action
                                    - raw_action
                                )
                            )
                        )

                        raise RuntimeError(
                            "Temporal Ensemble action_pred "
                            "alignment check failed. "
                            f"max_abs_diff="
                            f"{te_alignment_max_diff:.9g}"
                        )

                    # IMPORTANT:
                    # This is the absolute execution cursor at
                    # the START of the current policy query.
                    # It advances only after env.step reports
                    # how many controls were actually executed.
                    te_exec_start_this_query = (
                        cumulative_executed_steps.copy()
                    )

                    te_result = (
                        temporal_ensemble_current_chunk(
                            history=te_history,
                            current_action_pred=(
                                te_current_action_pred
                            ),
                            current_slice_start=(
                                te_action_slice_start
                            ),
                            current_exec_start=(
                                te_exec_start_this_query
                            ),
                            current_chunk_length=(
                                te_chunk_length
                            ),
                            beta=te_beta,
                        )
                    )

                    te_query_index = int(
                        te_result[
                            "current_query_index"
                        ]
                    )

                    # Existing persistent records must all come
                    # from strictly earlier policy queries.
                    if len(te_history) > 0:
                        te_latest_history_query = max(
                            int(
                                record[
                                    "query_index"
                                ]
                            )
                            for record in te_history
                        )

                        if (
                            te_latest_history_query
                            >= te_query_index
                        ):
                            raise RuntimeError(
                                "Temporal Ensemble history "
                                "contains a non-past query."
                            )

                    te_counts = np.asarray(
                        te_result["counts"]
                    )

                    if np.any(te_counts < 1):
                        raise RuntimeError(
                            "Temporal Ensemble produced an "
                            "action without a current-query "
                            "prediction."
                        )

                    # The current query has age zero and must be
                    # present for every output action. Negative
                    # ages are already rejected in the core.
                    if not np.all(
                        np.asarray(
                            te_result[
                                "source_age_min"
                            ]
                        ) == 0
                    ):
                        raise RuntimeError(
                            "Temporal Ensemble current-query "
                            "age audit failed."
                        )

                    te_action = np.asarray(
                        te_result["ensemble"],
                        dtype=raw_action.dtype,
                    )

                    if (
                        te_action.shape
                        != raw_action.shape
                    ):
                        raise RuntimeError(
                            "Temporal Ensemble output shape "
                            "differs from raw action."
                        )

                    # With no historical prediction, TE must be
                    # bitwise identical to the policy output.
                    if (
                        len(te_history) == 0
                        and not np.array_equal(
                            te_action,
                            raw_action,
                        )
                    ):
                        te_first_query_max_diff = float(
                            np.max(
                                np.abs(
                                    te_action
                                    - raw_action
                                )
                            )
                        )

                        raise RuntimeError(
                            "Temporal Ensemble first-query "
                            "identity failed. "
                            f"max_abs_diff="
                            f"{te_first_query_max_diff:.9g}"
                        )

                    # temporal_ensemble_current_chunk() includes
                    # the current prediction internally for this
                    # calculation but intentionally does NOT
                    # mutate persistent history. Append it once,
                    # and only once, after ensemble computation.
                    append_temporal_ensemble_record(
                        history=te_history,
                        action_pred=(
                            te_current_action_pred
                        ),
                        slice_start=(
                            te_action_slice_start
                        ),
                        exec_start=(
                            te_exec_start_this_query
                        ),
                        query_index=te_query_index,
                        max_history=te_max_history,
                    )

                    te_exec_start_history.append(
                        te_exec_start_this_query.copy()
                    )
                    te_candidate_count_history.append(
                        te_counts.copy()
                    )
                    te_source_age_max_history.append(
                        np.asarray(
                            te_result[
                                "source_age_max"
                            ]
                        ).copy()
                    )
                    te_query_index_history.append(
                        te_query_index
                    )

                    # TE is the complete baseline output sent
                    # downstream. BSOP and other smoothing are
                    # prohibited above when TE is active.
                    action = te_action.copy()

                if overlap_diag_enabled:
                    diagnostic_input_action = action.copy()

                    if "action_pred" not in np_action_dict:
                        raise RuntimeError(
                            "Policy result does not contain "
                            "'action_pred'."
                        )

                    diagnostic_action_pred = np.asarray(
                        np_action_dict["action_pred"]
                    )

                    if diagnostic_action_pred.ndim != 3:
                        raise RuntimeError(
                            "Expected action_pred shape "
                            "(batch, horizon, action_dim), got "
                            f"{diagnostic_action_pred.shape}."
                        )

                    if raw_action.ndim != 3:
                        raise RuntimeError(
                            "Expected action shape "
                            "(batch, steps, action_dim), got "
                            f"{raw_action.shape}."
                        )

                    diagnostic_pred_steps_only = bool(
                        getattr(
                            policy,
                            "pred_action_steps_only",
                            False,
                        )
                    )
                    diagnostic_n_obs_steps = int(
                        getattr(
                            policy,
                            "n_obs_steps",
                            self.n_obs_steps,
                        )
                    )
                    diagnostic_oa_convention = bool(
                        getattr(
                            policy,
                            "oa_step_convention",
                            False,
                        )
                    )

                    if diagnostic_pred_steps_only:
                        diagnostic_policy_slice_start = 0
                    else:
                        diagnostic_policy_slice_start = (
                            diagnostic_n_obs_steps
                            - int(diagnostic_oa_convention)
                        )

                    # Runner removes latency actions after
                    # predict_action returns.
                    diagnostic_action_slice_start = (
                        diagnostic_policy_slice_start
                        + int(self.n_latency_steps)
                    )
                    diagnostic_action_slice_end = (
                        diagnostic_action_slice_start
                        + int(raw_action.shape[1])
                    )

                    diagnostic_expected_action = (
                        diagnostic_action_pred[
                            :,
                            diagnostic_action_slice_start:
                            diagnostic_action_slice_end,
                        ]
                    )

                    if (
                        diagnostic_expected_action.shape
                        != raw_action.shape
                    ):
                        raise RuntimeError(
                            "Inferred action_pred slice has "
                            "the wrong shape: "
                            f"action_pred="
                            f"{diagnostic_action_pred.shape}, "
                            f"slice=["
                            f"{diagnostic_action_slice_start}:"
                            f"{diagnostic_action_slice_end}], "
                            f"expected="
                            f"{diagnostic_expected_action.shape}, "
                            f"returned={raw_action.shape}."
                        )

                    if not np.array_equal(
                        diagnostic_expected_action,
                        raw_action,
                    ):
                        diagnostic_max_diff = float(
                            np.max(
                                np.abs(
                                    diagnostic_expected_action
                                    - raw_action
                                )
                            )
                        )
                        raise RuntimeError(
                            "Returned action does not exactly "
                            "match its action_pred slice. "
                            f"max_abs_diff="
                            f"{diagnostic_max_diff:.9g}"
                        )

                    overlap_action_pred_history.append(
                        diagnostic_action_pred.copy()
                    )
                    overlap_slice_start_history.append(
                        int(diagnostic_action_slice_start)
                    )
                    overlap_horizon_history.append(
                        int(diagnostic_action_pred.shape[1])
                    )
                    overlap_chunk_length_history.append(
                        int(raw_action.shape[1])
                    )

                    if previous_action_pred is not None:
                        diagnostic_previous_overlap_start = (
                            int(previous_action_slice_start)
                            + int(previous_raw_action.shape[1])
                        )

                        diagnostic_available_previous = max(
                            0,
                            int(previous_action_pred.shape[1])
                            - diagnostic_previous_overlap_start,
                        )

                        diagnostic_overlap_length = min(
                            diagnostic_available_previous,
                            int(raw_action.shape[1]),
                        )

                        diagnostic_previous_overlap = (
                            previous_action_pred[
                                :,
                                diagnostic_previous_overlap_start:
                                diagnostic_previous_overlap_start
                                + diagnostic_overlap_length,
                            ].copy()
                        )
                        diagnostic_current_overlap = (
                            raw_action[
                                :,
                                :diagnostic_overlap_length,
                            ].copy()
                        )

                        if (
                            diagnostic_previous_overlap.shape
                            != diagnostic_current_overlap.shape
                        ):
                            raise RuntimeError(
                                "Aligned overlap shapes differ: "
                                f"previous="
                                f"{diagnostic_previous_overlap.shape}, "
                                f"current="
                                f"{diagnostic_current_overlap.shape}."
                            )

                        diagnostic_overlap_delta_l2 = (
                            np.linalg.norm(
                                diagnostic_current_overlap
                                - diagnostic_previous_overlap,
                                axis=-1,
                            )
                        )

                        diagnostic_raw_boundary_jump = (
                            np.linalg.norm(
                                raw_action[:, 0]
                                - previous_raw_action[:, -1],
                                axis=-1,
                            )
                        )

                        overlap_previous_prediction_history.append(
                            diagnostic_previous_overlap
                        )
                        overlap_current_prediction_history.append(
                            diagnostic_current_overlap
                        )
                        overlap_delta_l2_history.append(
                            diagnostic_overlap_delta_l2
                        )
                        overlap_length_history.append(
                            int(diagnostic_overlap_length)
                        )
                        overlap_raw_boundary_jump_history.append(
                            diagnostic_raw_boundary_jump.copy()
                        )

                    previous_action_pred = (
                        diagnostic_action_pred.copy()
                    )
                    previous_raw_action = raw_action.copy()
                    previous_action_slice_start = int(
                        diagnostic_action_slice_start
                    )

                    if not np.array_equal(
                        action,
                        diagnostic_input_action,
                    ):
                        raise RuntimeError(
                            "Overlap diagnostic modified "
                            "the action before filtering."
                        )

                # NaN means that no adaptive coefficient is applied
                # at that action position.
                trace_d = np.full(
                    action.shape[:2],
                    np.nan,
                    dtype=np.float64,
                )
                trace_alpha = np.full(
                    action.shape[:2],
                    np.nan,
                    dtype=np.float64,
                )

                if trace_enabled:
                    postprocess_start_ns = (
                        time.perf_counter_ns()
                    )

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
                        + int(self.n_latency_steps)
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

                # Action Smoothness Filter, ASF / Adaptive ASF
                # mode=fixed: use constant alpha.
                # mode=adaptive: reduce alpha when action changes abruptly.
                action_smooth_mode = os.environ.get("DP_ACTION_SMOOTH_MODE", "fixed")
                action_smooth_alpha = float(os.environ.get("DP_ACTION_SMOOTH_ALPHA", "1.0"))

                if (
                    (action_smooth_alpha < 1.0)
                    or action_smooth_mode in (
                        "sma",
                        "sg",
                    )
                ):
                    smoothed_action = action.copy()

                    if action_smooth_mode == "sma":
                        sma_window = int(os.environ.get("DP_ACTION_SMA_WINDOW", "3"))
                        sma_window = max(1, sma_window)
                        for t in range(smoothed_action.shape[1]):
                            start = max(0, t - sma_window + 1)
                            smoothed_action[:, t] = action[:, start:t+1].mean(axis=1)

                    elif action_smooth_mode == "sg":
                        sg_window = int(
                            os.environ.get(
                                "DP_ACTION_SG_WINDOW",
                                "5",
                            )
                        )
                        sg_polyorder = int(
                            os.environ.get(
                                "DP_ACTION_SG_POLYORDER",
                                "2",
                            )
                        )

                        if sg_window < 3:
                            raise RuntimeError(
                                "DP_ACTION_SG_WINDOW must "
                                "be at least 3."
                            )

                        if sg_window % 2 != 1:
                            raise RuntimeError(
                                "DP_ACTION_SG_WINDOW must "
                                "be odd."
                            )

                        if sg_polyorder < 0:
                            raise RuntimeError(
                                "DP_ACTION_SG_POLYORDER "
                                "must be non-negative."
                            )

                        if sg_polyorder >= sg_window:
                            raise RuntimeError(
                                "DP_ACTION_SG_POLYORDER "
                                "must be smaller than "
                                "DP_ACTION_SG_WINDOW."
                            )

                        if sg_window > action.shape[1]:
                            raise RuntimeError(
                                "DP_ACTION_SG_WINDOW exceeds "
                                "the current action chunk "
                                f"length: window={sg_window}, "
                                f"chunk={action.shape[1]}."
                            )

                        smoothed_action = savgol_filter(
                            action,
                            window_length=sg_window,
                            polyorder=sg_polyorder,
                            axis=1,
                            mode="interp",
                        ).astype(
                            action.dtype,
                            copy=False,
                        )

                    elif action_smooth_mode == "adaptive":
                        alpha_min = float(os.environ.get("DP_ACTION_SMOOTH_ALPHA_MIN", "0.45"))
                        alpha_max = float(os.environ.get("DP_ACTION_SMOOTH_ALPHA_MAX", str(action_smooth_alpha)))
                        threshold = float(os.environ.get("DP_ACTION_SMOOTH_THRESHOLD", "10.0"))
                        gain = float(os.environ.get("DP_ACTION_SMOOTH_GAIN", "0.25"))

                        for t in range(1, smoothed_action.shape[1]):
                            delta = np.linalg.norm(
                                action[:, t] - smoothed_action[:, t-1],
                                axis=-1,
                                keepdims=True
                            )
                            excess = np.maximum(delta - threshold, 0.0) / (threshold + 1e-6)
                            adaptive_alpha = action_smooth_alpha - gain * excess
                            adaptive_alpha = np.clip(
                                adaptive_alpha,
                                alpha_min,
                                alpha_max,
                            )

                            if trace_enabled:
                                trace_d[:, t] = delta[:, 0]
                                trace_alpha[:, t] = (
                                    adaptive_alpha[:, 0]
                                )

                            smoothed_action[:, t] = (
                                adaptive_alpha * action[:, t]
                                + (1.0 - adaptive_alpha) * smoothed_action[:, t-1]
                            )
                    else:
                        for t in range(1, smoothed_action.shape[1]):
                            smoothed_action[:, t] = (
                                action_smooth_alpha * action[:, t]
                                + (1.0 - action_smooth_alpha) * smoothed_action[:, t-1]
                            )

                    action = smoothed_action

                if overlap_diag_enabled:
                    diagnostic_filtered_action = (
                        action.copy()
                    )

                    if previous_filtered_action is not None:
                        diagnostic_filtered_boundary_jump = (
                            np.linalg.norm(
                                action[:, 0]
                                - previous_filtered_action[:, -1],
                                axis=-1,
                            )
                        )
                        overlap_filtered_boundary_jump_history.append(
                            diagnostic_filtered_boundary_jump.copy()
                        )

                    previous_filtered_action = action.copy()

                    if not np.array_equal(
                        action,
                        diagnostic_filtered_action,
                    ):
                        raise RuntimeError(
                            "Boundary diagnostic modified "
                            "the action sent to env.step."
                        )

                if trace_enabled:
                    postprocess_end_ns = (
                        time.perf_counter_ns()
                    )
                    decision_end_ns = postprocess_end_ns

                    batch_size = int(action.shape[0])

                    policy_ms = (
                        policy_end_ns
                        - policy_start_ns
                    ) / 1.0e6

                    postprocess_ms = (
                        postprocess_end_ns
                        - postprocess_start_ns
                    ) / 1.0e6

                    decision_ms = (
                        decision_end_ns
                        - decision_start_ns
                    ) / 1.0e6

                    raw_action_history.append(
                        raw_action.copy()
                    )
                    d_history.append(
                        trace_d.copy()
                    )
                    alpha_history.append(
                        trace_alpha.copy()
                    )

                    policy_time_history.append(
                        np.full(
                            (batch_size,),
                            policy_ms,
                            dtype=np.float64,
                        )
                    )
                    postprocess_time_history.append(
                        np.full(
                            (batch_size,),
                            postprocess_ms,
                            dtype=np.float64,
                        )
                    )
                    decision_time_history.append(
                        np.full(
                            (batch_size,),
                            decision_ms,
                            dtype=np.float64,
                        )
                    )
                    chunk_length_history.append(
                        int(action.shape[1])
                    )

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

                action_history.append(action.copy())

                # Step the vector environment. The runner has
                # already recorded the complete planned chunk,
                # but MultiStepWrapper may stop partway through
                # it after success or time-limit termination.
                obs, reward, done, info = env.step(action)

                reward_histories_after_step = env.call(
                    "get_attr",
                    "reward",
                )

                current_executed_steps = np.asarray(
                    [
                        len(reward_history)
                        for reward_history
                        in reward_histories_after_step
                    ],
                    dtype=np.int64,
                )

                if current_executed_steps.shape != (
                    n_envs,
                ):
                    raise RuntimeError(
                        "Unexpected cumulative execution "
                        "counter shape: "
                        f"{current_executed_steps.shape}, "
                        f"expected {(n_envs,)}."
                    )

                executed_steps_this_query = (
                    current_executed_steps
                    - cumulative_executed_steps
                )

                planned_steps_this_query = int(
                    action.shape[1]
                )

                if np.any(
                    executed_steps_this_query < 0
                ):
                    raise RuntimeError(
                        "Cumulative executed-action count "
                        "decreased."
                    )

                if np.any(
                    executed_steps_this_query
                    > planned_steps_this_query
                ):
                    raise RuntimeError(
                        "An environment executed more actions "
                        "than were supplied in the chunk: "
                        f"executed="
                        f"{executed_steps_this_query}, "
                        f"planned="
                        f"{planned_steps_this_query}."
                    )

                planned_steps_per_query_history.append(
                    np.full(
                        (n_envs,),
                        planned_steps_this_query,
                        dtype=np.int64,
                    )
                )
                executed_steps_per_query_history.append(
                    executed_steps_this_query.copy()
                )

                for execution_env_index in range(
                    n_envs
                ):
                    execution_count = int(
                        executed_steps_this_query[
                            execution_env_index
                        ]
                    )

                    if execution_count > 0:
                        executed_action_parts[
                            execution_env_index
                        ].append(
                            action[
                                execution_env_index,
                                :execution_count,
                            ].copy()
                        )

                if te_enabled:
                    if te_exec_start_this_query is None:
                        raise RuntimeError(
                            "Temporal Ensemble execution "
                            "start was not initialized."
                        )

                    # Absolute next-query cursor must be:
                    #
                    # start_of_this_query
                    # + ACTUALLY executed controls
                    #
                    # and never start + planned chunk length.
                    te_expected_next_cursor = (
                        te_exec_start_this_query
                        + executed_steps_this_query
                    )

                    if not np.array_equal(
                        te_expected_next_cursor,
                        current_executed_steps,
                    ):
                        raise RuntimeError(
                            "Temporal Ensemble actual-execution "
                            "cursor accounting failed: "
                            f"expected="
                            f"{te_expected_next_cursor}, "
                            f"actual="
                            f"{current_executed_steps}."
                        )

                    te_executed_steps_history.append(
                        executed_steps_this_query.copy()
                    )

                cumulative_executed_steps = (
                    current_executed_steps
                )

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            if te_enabled:
                if len(te_candidate_count_history) == 0:
                    raise RuntimeError(
                        "Temporal Ensemble recorded no "
                        "policy queries."
                    )

                if (
                    len(te_exec_start_history)
                    != len(
                        te_executed_steps_history
                    )
                ):
                    raise RuntimeError(
                        "Temporal Ensemble query/execution "
                        "audit lengths differ."
                    )

                te_all_counts = np.concatenate(
                    [
                        x.reshape(-1)
                        for x
                        in te_candidate_count_history
                    ]
                )

                te_all_max_ages = np.concatenate(
                    [
                        x.reshape(-1)
                        for x
                        in te_source_age_max_history
                    ]
                )

                print(
                    "[temporal-ensemble] "
                    f"beta={te_beta:g}, "
                    f"queries="
                    f"{len(te_query_index_history)}, "
                    f"candidate_count="
                    f"[{int(np.min(te_all_counts))},"
                    f"{int(np.max(te_all_counts))}], "
                    f"max_source_age="
                    f"{int(np.max(te_all_max_ages))}, "
                    "actual_cursor=PASS"
                )

            if overlap_diag_enabled:
                if len(overlap_action_pred_history) == 0:
                    raise RuntimeError(
                        "Overlap diagnostic recorded no "
                        "policy predictions."
                    )

                if (
                    len(overlap_raw_boundary_jump_history)
                    != len(
                        overlap_filtered_boundary_jump_history
                    )
                ):
                    raise RuntimeError(
                        "Raw and filtered boundary counts "
                        "do not match."
                    )

                if (
                    len(overlap_length_history)
                    != len(
                        overlap_raw_boundary_jump_history
                    )
                ):
                    raise RuntimeError(
                        "Overlap and boundary counts "
                        "do not match."
                    )

                diagnostic_unique_lengths = sorted(
                    set(overlap_length_history)
                )

                if len(diagnostic_unique_lengths) > 1:
                    raise RuntimeError(
                        "Overlap length changed within an "
                        "episode: "
                        f"{diagnostic_unique_lengths}"
                    )

                diagnostic_action_pred_stack = np.stack(
                    overlap_action_pred_history,
                    axis=1,
                )

                diagnostic_batch_size = int(
                    diagnostic_action_pred_stack.shape[0]
                )
                diagnostic_action_dim = int(
                    diagnostic_action_pred_stack.shape[-1]
                )

                if len(
                    overlap_previous_prediction_history
                ) > 0:
                    diagnostic_previous_overlap_stack = (
                        np.stack(
                            overlap_previous_prediction_history,
                            axis=1,
                        )
                    )
                    diagnostic_current_overlap_stack = (
                        np.stack(
                            overlap_current_prediction_history,
                            axis=1,
                        )
                    )
                    diagnostic_delta_l2_stack = np.stack(
                        overlap_delta_l2_history,
                        axis=1,
                    )
                    diagnostic_raw_jump_stack = np.stack(
                        overlap_raw_boundary_jump_history,
                        axis=1,
                    )
                    diagnostic_filtered_jump_stack = (
                        np.stack(
                            overlap_filtered_boundary_jump_history,
                            axis=1,
                        )
                    )
                else:
                    diagnostic_previous_overlap_stack = (
                        np.empty(
                            (
                                diagnostic_batch_size,
                                0,
                                0,
                                diagnostic_action_dim,
                            ),
                            dtype=np.float32,
                        )
                    )
                    diagnostic_current_overlap_stack = (
                        diagnostic_previous_overlap_stack.copy()
                    )
                    diagnostic_delta_l2_stack = np.empty(
                        (
                            diagnostic_batch_size,
                            0,
                            0,
                        ),
                        dtype=np.float32,
                    )
                    diagnostic_raw_jump_stack = np.empty(
                        (
                            diagnostic_batch_size,
                            0,
                        ),
                        dtype=np.float32,
                    )
                    diagnostic_filtered_jump_stack = (
                        diagnostic_raw_jump_stack.copy()
                    )

                for diagnostic_local_index in range(
                    this_n_active_envs
                ):
                    diagnostic_global_index = (
                        start + diagnostic_local_index
                    )
                    diagnostic_seed = int(
                        self.env_seeds[
                            diagnostic_global_index
                        ]
                    )
                    diagnostic_prefix = (
                        self.env_prefixs[
                            diagnostic_global_index
                        ]
                    )
                    diagnostic_prefix_name = (
                        diagnostic_prefix
                        .strip("/")
                        .replace("/", "_")
                    )

                    if not diagnostic_prefix_name:
                        diagnostic_prefix_name = "unknown"

                    diagnostic_path = trace_root / (
                        f"{diagnostic_prefix_name}"
                        f"_seed_{diagnostic_seed}"
                        "_overlap_diag.npz"
                    )

                    np.savez_compressed(
                        diagnostic_path,
                        diag_version=np.asarray(
                            "overlap_diag_v2"
                        ),
                        seed=np.int64(
                            diagnostic_seed
                        ),
                        prefix=np.asarray(
                            diagnostic_prefix
                        ),
                        method=np.asarray(
                            trace_method
                        ),
                        action_pred=(
                            diagnostic_action_pred_stack[
                                diagnostic_local_index
                            ]
                        ),
                        previous_overlap_prediction=(
                            diagnostic_previous_overlap_stack[
                                diagnostic_local_index
                            ]
                        ),
                        current_overlap_prediction=(
                            diagnostic_current_overlap_stack[
                                diagnostic_local_index
                            ]
                        ),
                        overlap_prediction_delta_l2=(
                            diagnostic_delta_l2_stack[
                                diagnostic_local_index
                            ]
                        ),
                        raw_boundary_jump_l2=(
                            diagnostic_raw_jump_stack[
                                diagnostic_local_index
                            ]
                        ),
                        filtered_boundary_jump_l2=(
                            diagnostic_filtered_jump_stack[
                                diagnostic_local_index
                            ]
                        ),
                        action_slice_start=np.asarray(
                            overlap_slice_start_history,
                            dtype=np.int64,
                        ),
                        action_pred_horizon=np.asarray(
                            overlap_horizon_history,
                            dtype=np.int64,
                        ),
                        executed_chunk_length=np.asarray(
                            overlap_chunk_length_history,
                            dtype=np.int64,
                        ),
                        overlap_length=np.asarray(
                            overlap_length_history,
                            dtype=np.int64,
                        ),
                        n_obs_steps=np.int64(
                            getattr(
                                policy,
                                "n_obs_steps",
                                self.n_obs_steps,
                            )
                        ),
                        n_action_steps=np.int64(
                            getattr(
                                policy,
                                "n_action_steps",
                                self.n_action_steps,
                            )
                        ),
                        n_latency_steps=np.int64(
                            self.n_latency_steps
                        ),
                        pred_action_steps_only=np.bool_(
                            getattr(
                                policy,
                                "pred_action_steps_only",
                                False,
                            )
                        ),
                        oa_step_convention=np.bool_(
                            getattr(
                                policy,
                                "oa_step_convention",
                                False,
                            )
                        ),
                    )

                    print(
                        "[overlap-diag] saved "
                        f"{diagnostic_path}"
                    )

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            if len(action_history) > 0:
                chunk_actions = np.concatenate(
                    action_history,
                    axis=1,
                )
                local_actions = chunk_actions[
                    this_local_slice
                ]
                all_action_trajs[
                    this_global_slice
                ] = local_actions

                if len(
                    planned_steps_per_query_history
                ) != len(action_history):
                    raise RuntimeError(
                        "Planned-step history and action "
                        "query count differ."
                    )

                if len(
                    executed_steps_per_query_history
                ) != len(action_history):
                    raise RuntimeError(
                        "Executed-step history and action "
                        "query count differ."
                    )

                planned_steps_matrix = np.stack(
                    planned_steps_per_query_history,
                    axis=1,
                )
                executed_steps_matrix = np.stack(
                    executed_steps_per_query_history,
                    axis=1,
                )

                if (
                    planned_steps_matrix.shape
                    != executed_steps_matrix.shape
                ):
                    raise RuntimeError(
                        "Planned/executed query matrices "
                        "have different shapes: "
                        f"planned="
                        f"{planned_steps_matrix.shape}, "
                        f"executed="
                        f"{executed_steps_matrix.shape}."
                    )

                for execution_local_index, (
                    execution_global_index
                ) in enumerate(range(start, end)):
                    planned_per_query = (
                        planned_steps_matrix[
                            execution_local_index
                        ].copy()
                    )
                    executed_per_query = (
                        executed_steps_matrix[
                            execution_local_index
                        ].copy()
                    )

                    execution_parts = (
                        executed_action_parts[
                            execution_local_index
                        ]
                    )

                    if len(execution_parts) > 0:
                        actual_executed_action = (
                            np.concatenate(
                                execution_parts,
                                axis=0,
                            )
                        )
                    else:
                        actual_executed_action = np.empty(
                            (
                                0,
                                int(
                                    local_actions.shape[-1]
                                ),
                            ),
                            dtype=local_actions.dtype,
                        )

                    expected_actual_count = int(
                        np.sum(executed_per_query)
                    )

                    if (
                        actual_executed_action.shape[0]
                        != expected_actual_count
                    ):
                        raise RuntimeError(
                            "Executed-action trajectory "
                            "length does not match the "
                            "per-query execution counters: "
                            f"trajectory="
                            f"{actual_executed_action.shape[0]}, "
                            f"counter="
                            f"{expected_actual_count}."
                        )

                    all_planned_steps_per_query[
                        execution_global_index
                    ] = planned_per_query
                    all_executed_steps_per_query[
                        execution_global_index
                    ] = executed_per_query
                    all_actual_executed_action_trajs[
                        execution_global_index
                    ] = actual_executed_action.copy()

                if trace_enabled:
                    chunk_raw_actions = np.concatenate(
                        raw_action_history,
                        axis=1,
                    )
                    chunk_d = np.concatenate(
                        d_history,
                        axis=1,
                    )
                    chunk_alpha = np.concatenate(
                        alpha_history,
                        axis=1,
                    )

                    chunk_policy_ms = np.stack(
                        policy_time_history,
                        axis=1,
                    )
                    chunk_postprocess_ms = np.stack(
                        postprocess_time_history,
                        axis=1,
                    )
                    chunk_decision_ms = np.stack(
                        decision_time_history,
                        axis=1,
                    )

                    local_raw_actions = (
                        chunk_raw_actions[
                            this_local_slice
                        ]
                    )
                    local_d = chunk_d[
                        this_local_slice
                    ]
                    local_alpha = chunk_alpha[
                        this_local_slice
                    ]
                    local_policy_ms = (
                        chunk_policy_ms[
                            this_local_slice
                        ]
                    )
                    local_postprocess_ms = (
                        chunk_postprocess_ms[
                            this_local_slice
                        ]
                    )
                    local_decision_ms = (
                        chunk_decision_ms[
                            this_local_slice
                        ]
                    )

                    all_raw_action_trajs[
                        this_global_slice
                    ] = local_raw_actions
                    all_d_trajs[
                        this_global_slice
                    ] = local_d
                    all_alpha_trajs[
                        this_global_slice
                    ] = local_alpha
                    all_policy_times_ms[
                        this_global_slice
                    ] = local_policy_ms
                    all_postprocess_times_ms[
                        this_global_slice
                    ] = local_postprocess_ms
                    all_decision_times_ms[
                        this_global_slice
                    ] = local_decision_ms

                    chunk_starts = np.cumsum(
                        np.asarray(
                            [0]
                            + chunk_length_history[:-1],
                            dtype=np.int64,
                        )
                    )

                    number_local = int(
                        local_actions.shape[0]
                    )
                    all_chunk_starts[
                        this_global_slice
                    ] = [
                        chunk_starts.copy()
                        for _ in range(number_local)
                    ]
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        action_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
        log_data = dict()

        def _mean_l2(x):
            if x is None or len(x) == 0:
                return 0.0
            x = np.asarray(x, dtype=np.float32)
            return float(np.linalg.norm(x, axis=-1).mean())
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

            # action smoothness metrics
            action_traj = all_action_trajs[i]
            if action_traj is not None:
                action_traj = np.asarray(action_traj, dtype=np.float32)
                action_l2_mean = _mean_l2(action_traj)
                action_delta_l2_mean = _mean_l2(np.diff(action_traj, n=1, axis=0))
                action_acc_l2_mean = _mean_l2(np.diff(action_traj, n=2, axis=0))
                action_jerk_l2_mean = _mean_l2(np.diff(action_traj, n=3, axis=0))
                action_smoothness_score = float(1.0 / (1.0 + action_jerk_l2_mean))

                log_data[prefix+f'action_l2_mean_{seed}'] = action_l2_mean
                log_data[prefix+f'action_delta_l2_mean_{seed}'] = action_delta_l2_mean
                log_data[prefix+f'action_acc_l2_mean_{seed}'] = action_acc_l2_mean
                log_data[prefix+f'action_jerk_l2_mean_{seed}'] = action_jerk_l2_mean
                log_data[prefix+f'action_smoothness_score_{seed}'] = action_smoothness_score

                action_metrics[prefix]['action_l2_mean'].append(action_l2_mean)
                action_metrics[prefix]['action_delta_l2_mean'].append(action_delta_l2_mean)
                action_metrics[prefix]['action_acc_l2_mean'].append(action_acc_l2_mean)
                action_metrics[prefix]['action_jerk_l2_mean'].append(action_jerk_l2_mean)
                action_metrics[prefix]['action_smoothness_score'].append(action_smoothness_score)

            if trace_enabled:
                raw_action_traj = (
                    all_raw_action_trajs[i]
                )
                d_traj = all_d_trajs[i]
                alpha_traj = all_alpha_trajs[i]
                chunk_starts = all_chunk_starts[i]
                policy_times_ms = (
                    all_policy_times_ms[i]
                )
                postprocess_times_ms = (
                    all_postprocess_times_ms[i]
                )
                decision_times_ms = (
                    all_decision_times_ms[i]
                )

                planned_steps_per_query = (
                    all_planned_steps_per_query[i]
                )
                executed_steps_per_query = (
                    all_executed_steps_per_query[i]
                )
                actual_executed_action_traj = (
                    all_actual_executed_action_trajs[i]
                )

                if planned_steps_per_query is None:
                    raise RuntimeError(
                        "Trace enabled but planned-step "
                        f"history is missing for seed {seed}."
                    )

                if executed_steps_per_query is None:
                    raise RuntimeError(
                        "Trace enabled but executed-step "
                        f"history is missing for seed {seed}."
                    )

                if actual_executed_action_traj is None:
                    raise RuntimeError(
                        "Trace enabled but actual executed "
                        f"actions are missing for seed {seed}."
                    )

                planned_steps_per_query = np.asarray(
                    planned_steps_per_query,
                    dtype=np.int64,
                )
                executed_steps_per_query = np.asarray(
                    executed_steps_per_query,
                    dtype=np.int64,
                )
                actual_executed_action_traj = np.asarray(
                    actual_executed_action_traj,
                    dtype=np.float32,
                )

                if (
                    planned_steps_per_query.shape
                    != executed_steps_per_query.shape
                ):
                    raise RuntimeError(
                        "Trace planned/executed query shapes "
                        f"differ for seed {seed}: "
                        f"planned="
                        f"{planned_steps_per_query.shape}, "
                        f"executed="
                        f"{executed_steps_per_query.shape}."
                    )

                reward_history = np.asarray(
                    all_rewards[i],
                    dtype=np.float64,
                )

                actual_executed_steps = int(
                    np.sum(executed_steps_per_query)
                )
                planned_action_steps_total = int(
                    np.sum(planned_steps_per_query)
                )
                unexecuted_steps_per_query = (
                    planned_steps_per_query
                    - executed_steps_per_query
                )
                unexecuted_action_steps_total = int(
                    np.sum(
                        unexecuted_steps_per_query
                    )
                )

                if (
                    reward_history.shape[0]
                    != actual_executed_steps
                ):
                    raise RuntimeError(
                        "Reward-history length does not "
                        "match actual executed-action count "
                        f"for seed {seed}: "
                        f"rewards={reward_history.shape[0]}, "
                        f"executed={actual_executed_steps}."
                    )

                if (
                    actual_executed_action_traj.shape[0]
                    != actual_executed_steps
                ):
                    raise RuntimeError(
                        "Executed-action array length does "
                        "not match its counter for seed "
                        f"{seed}."
                    )

                if (
                    np.asarray(action_traj).shape[0]
                    != planned_action_steps_total
                ):
                    raise RuntimeError(
                        "Legacy planned-action trajectory "
                        "length does not match planned-step "
                        f"counter for seed {seed}."
                    )

                if raw_action_traj is None:
                    raise RuntimeError(
                        "Trace enabled but raw action "
                        f"trajectory is missing for seed {seed}"
                    )

                prefix_name = (
                    prefix.strip("/")
                    .replace("/", "_")
                )
                if not prefix_name:
                    prefix_name = "unknown"

                trace_path = trace_root / (
                    f"{prefix_name}_seed_{seed}.npz"
                )

                np.savez_compressed(
                    trace_path,
                    seed=np.int64(seed),
                    prefix=np.asarray(prefix),
                    method=np.asarray(trace_method),
                    score=np.float64(max_reward),
                    warmup_queries=np.int64(
                        trace_warmup_queries
                    ),
                    raw_action=np.asarray(
                        raw_action_traj,
                        dtype=np.float32,
                    ),
                    filtered_action=np.asarray(
                        action_traj,
                        dtype=np.float32,
                    ),

                    # Strict actual-execution fields.
                    execution_trace_version=np.asarray(
                        "actual_execution_trace_v1"
                    ),
                    executed_action=np.asarray(
                        actual_executed_action_traj,
                        dtype=np.float32,
                    ),
                    planned_steps_per_query=np.asarray(
                        planned_steps_per_query,
                        dtype=np.int64,
                    ),
                    executed_steps_per_query=np.asarray(
                        executed_steps_per_query,
                        dtype=np.int64,
                    ),
                    unexecuted_steps_per_query=np.asarray(
                        unexecuted_steps_per_query,
                        dtype=np.int64,
                    ),
                    cumulative_executed_steps=np.cumsum(
                        executed_steps_per_query,
                        dtype=np.int64,
                    ),
                    actual_executed_steps=np.int64(
                        actual_executed_steps
                    ),
                    planned_action_steps_total=np.int64(
                        planned_action_steps_total
                    ),
                    unexecuted_action_steps_total=np.int64(
                        unexecuted_action_steps_total
                    ),
                    final_chunk_executed_steps=np.int64(
                        executed_steps_per_query[-1]
                    ),
                    final_chunk_unexecuted_steps=np.int64(
                        unexecuted_steps_per_query[-1]
                    ),
                    reward_history=np.asarray(
                        reward_history,
                        dtype=np.float64,
                    ),
                    max_episode_steps=np.int64(
                        self.max_steps
                    ),
                    terminated_before_time_limit=np.bool_(
                        actual_executed_steps
                        < int(self.max_steps)
                    ),
                    reached_time_limit=np.bool_(
                        actual_executed_steps
                        >= int(self.max_steps)
                    ),

                    d=np.asarray(
                        d_traj,
                        dtype=np.float64,
                    ),
                    alpha=np.asarray(
                        alpha_traj,
                        dtype=np.float64,
                    ),
                    chunk_starts=np.asarray(
                        chunk_starts,
                        dtype=np.int64,
                    ),
                    policy_inference_ms=np.asarray(
                        policy_times_ms,
                        dtype=np.float64,
                    ),
                    postprocess_ms=np.asarray(
                        postprocess_times_ms,
                        dtype=np.float64,
                    ),
                    decision_loop_ms=np.asarray(
                        decision_times_ms,
                        dtype=np.float64,
                    ),
                )

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

        for prefix, metric_dict in action_metrics.items():
            for metric_name, values in metric_dict.items():
                log_data[prefix+metric_name] = float(np.mean(values))
                log_data[prefix+metric_name+'_std'] = float(np.std(values))

        return log_data
