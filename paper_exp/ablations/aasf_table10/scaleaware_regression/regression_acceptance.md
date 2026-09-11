# Scale-aware AASF Regression Acceptance

**Status:** PASS

## Frozen threshold rule

`tau_task = kappa * median_within_episode_step_norm`

- kappa: `1.2403473459`
- Push-T median step norm: `8.0622577483`
- Computed Push-T threshold: `10.0000000000`

## Regression results

| Metric | Result |
|---|---:|
| Episodes | 50 |
| Mean Score | 0.9316082842 |
| SR@0.90 | 0.92 |
| SR@0.95 | 0.92 |
| Fail<0.90 | 4 |
| Action Delta | 8.9468791533 |
| Action Acceleration | 6.6355705929 |
| Action Jerk | 10.8013496637 |
| Smoothness Score | 0.1003194702 |

## Conclusion

The scale-aware implementation exactly reproduces the frozen
fixed-threshold AASF result on Push-T.

The implementation is accepted for second-task adaptation.
