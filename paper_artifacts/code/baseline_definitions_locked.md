# Locked Baseline Definitions

All smoothing baselines are implemented as inference-time action post-processing in:

diffusion_policy/env_runner/pusht_keypoints_runner.py

The filter is applied after policy.predict_action() and before env.step(action).
The filter operates within each predicted action chunk. No smoothing state is carried across different policy queries/chunks.

## Original Diffusion Policy

No smoothing is applied.

Environment setting:

DP_ACTION_SMOOTH_ALPHA=1.0

Since action_smooth_alpha is not smaller than 1.0 and mode is not sma, the smoothing block is skipped.

## SMA-3

Causal simple moving average with window size 3.

Environment settings:

DP_ACTION_SMOOTH_MODE=sma
DP_ACTION_SMA_WINDOW=3

For each timestep t in the predicted action chunk:

start = max(0, t - 3 + 1)

a_smooth[t] = mean(a[start:t+1])

The average uses only current and previous actions within the same chunk.

## SMA-5

Causal simple moving average with window size 5.

Environment settings:

DP_ACTION_SMOOTH_MODE=sma
DP_ACTION_SMA_WINDOW=5

For each timestep t in the predicted action chunk:

start = max(0, t - 5 + 1)

a_smooth[t] = mean(a[start:t+1])

The average uses only current and previous actions within the same chunk.

## Fixed EMA

Fixed exponential moving average.

Environment settings:

DP_ACTION_SMOOTH_MODE=fixed
DP_ACTION_SMOOTH_ALPHA=0.7

Initialization:

a_smooth[0] = a[0]

For t >= 1:

a_smooth[t] = alpha * a[t] + (1 - alpha) * a_smooth[t-1]

where alpha = 0.7.

A smaller alpha means stronger smoothing.

## AASF

Adaptive Action Smoothness Filter.

Environment settings:

DP_ACTION_SMOOTH_MODE=adaptive
DP_ACTION_SMOOTH_ALPHA=0.8
DP_ACTION_SMOOTH_THRESHOLD=10.0
DP_ACTION_SMOOTH_GAIN=0.25

Code defaults:

DP_ACTION_SMOOTH_ALPHA_MIN=0.45
DP_ACTION_SMOOTH_ALPHA_MAX=DP_ACTION_SMOOTH_ALPHA

Initialization:

a_smooth[0] = a[0]

For t >= 1:

d_t = || a[t] - a_smooth[t-1] ||_2

excess_t = max(d_t - threshold, 0) / (threshold + 1e-6)

alpha_t = clip(alpha - gain * excess_t, alpha_min, alpha_max)

a_smooth[t] = alpha_t * a[t] + (1 - alpha_t) * a_smooth[t-1]

When action changes abruptly, alpha_t decreases, which increases reliance on the previous smoothed action and suppresses abrupt action jumps.

## Important Implementation Notes

1. All filters are causal within the predicted action chunk.
2. No filter state is carried across chunks.
3. The implementation is inference-time post-processing, not a training-time loss or network module.
4. AASF must be run with DP_ACTION_SMOOTH_ALPHA < 1.0; otherwise, the current guard condition may skip the smoothing block.
