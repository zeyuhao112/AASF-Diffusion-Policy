import wandb
import numpy as np
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

                # Action Smoothness Filter, ASF / Adaptive ASF
                # mode=fixed: use constant alpha.
                # mode=adaptive: reduce alpha when action changes abruptly.
                action_smooth_mode = os.environ.get("DP_ACTION_SMOOTH_MODE", "fixed")
                action_smooth_alpha = float(os.environ.get("DP_ACTION_SMOOTH_ALPHA", "1.0"))

                if (action_smooth_alpha < 1.0) or (action_smooth_mode == "sma"):
                    smoothed_action = action.copy()

                    if action_smooth_mode == "sma":
                        sma_window = int(os.environ.get("DP_ACTION_SMA_WINDOW", "3"))
                        sma_window = max(1, sma_window)
                        for t in range(smoothed_action.shape[1]):
                            start = max(0, t - sma_window + 1)
                            smoothed_action[:, t] = action[:, start:t+1].mean(axis=1)

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

                action_history.append(action.copy())

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

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
