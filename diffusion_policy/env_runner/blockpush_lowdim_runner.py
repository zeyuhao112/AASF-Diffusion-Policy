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

        # plan for rollout
        n_envs = len(self.env_fns)
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

                # step env
                obs, reward, done, info = env.step(action)

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

                    np.savez_compressed(
                        trace_path,
                        seed=np.asarray(
                            seed,
                            dtype=np.int64,
                        ),
                        mode=np.asarray(
                            action_smooth_mode
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
