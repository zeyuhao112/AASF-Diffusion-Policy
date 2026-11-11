# AASF–BSOP results summary

## Metric convention

Positive percentage reductions mean smoother commands. D1, D2, and D3 are
finite-difference command metrics. C1, C2, and C3 divide them by `dt`, `dt²`,
and `dt³`; percentage reductions are therefore unchanged. These are
command-space proxies, not direct measurements of physical acceleration,
torque, or mechanical jerk.

## Seven-condition BSOP evaluation

Each condition uses 50 paired formal evaluation seeds and the actually
executed command stream at 10 Hz.

| Condition | BSOP−Original score | Boundary D1 reduction | C2 reduction | C3 reduction |
|---|---:|---:|---:|---:|
| Push-T lowdim seed42 | +0.001740 | 20.141429% | 7.537254% | 14.158802% |
| Push-T lowdim seed43 | −0.000719 | 15.898807% | 5.793048% | 12.499793% |
| Push-T lowdim seed44 | −0.023264 | 13.347558% | 5.837820% | 11.746279% |
| BlockPush lowdim | −0.000400 | 16.859402% | 4.001281% | 8.004071% |
| Push-T image seed42 | +0.023939 | 15.739123% | 6.937074% | 12.013036% |
| Push-T image seed43 | −0.022441 | 13.990875% | 4.620270% | 9.918027% |
| Push-T image seed44 | +0.012645 | 12.696782% | 6.371628% | 10.258626% |

Bootstrap intervals, per-seed values, Holm-adjusted tests, and the canonical
cross-check are in `paper_exp/tables/bsop_time_normalized_7cond_v1/`.

## Original four-method Push-T condition

For the low-dimensional seed-42 checkpoint, the BSOP+AASF mechanism audit
contains 1,504 boundaries, 69.08% activation, 14.5386% pooled same-boundary
reduction, and a per-seed 95% bootstrap interval of [12.9596%, 16.3118%]. All
50 paired seeds improve and all geometric-constraint violation counts are zero.

## AASF image transfer

Without retuning, AASF reduces D3/C3 by 16.14%, 15.21%, and 21.63% on the
seed-42, seed-43, and seed-44 Push-T image checkpoints, respectively. The paired
score differences are −0.0530, −0.0224, and +0.0392; corresponding confidence
intervals are provided in the frozen three-condition table.

## Baselines

- Savitzky–Golay (`window=5`, `polyorder=3`) has a confirmatory D3/C3 reduction
  of 15.5107% with 95% CI [14.0132%, 17.1325%]. Its paired score difference is
  −0.000970 with an interval containing zero.
- Temporal Ensemble (`beta=2.0`) has a confirmatory D3/C3 point estimate of
  −3.4841% with 95% CI [−8.8287%, 1.3752%]. The frozen study does not support a
  reliable improvement claim.

## Post-processing timing

The final CPU microbenchmark reports mean per-query times of 73.54 μs for
BSOP, 180.63 μs for AASF, 272.70 μs for BSOP+AASF, and 97.66 μs for the full
TE-beta2 path. These isolated post-processing timings must not be interpreted as
end-to-end policy acceleration.

## Franka simulated replay

In the frozen MuJoCo Franka replay, Original→BSOP reduces boundary-window
torque-rate RMS by 16.23% (95% CI [13.19%, 19.23%]) and boundary-window
end-effector jerk RMS by 17.64% (95% CI [13.87%, 21.71%]). This is simulated
dynamics replay, not real-hardware validation.

## SO-101 held-out offline audit

On 10 recorded held-out episodes, teacher-forced offline BSOP projection reduces
the pooled normalized boundary discrepancy from 0.190532388 to 0.163906747:
13.974339%, with an episode-cluster 95% CI of [11.504938%, 16.098201%]. It is
active on 167/263 boundaries and produces zero recorded constraint violations.

The mean applied interpolation coefficient is 0.160965 over all boundaries and
0.253496 over active boundaries. This experiment does not constitute online
closed-loop deployment or physical vibration validation.
