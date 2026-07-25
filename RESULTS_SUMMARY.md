# Final AASF Results Summary

## Experimental design

- Task: Push-T keypoint manipulation.
- Comparison: Original Diffusion Policy versus AASF.
- Evaluation design: 100 common-random-number paired episodes.
- Paired evaluation seeds: 4300000–4300099.
- AASF parameters:
  - alpha_min = 0.45
  - alpha_max = 0.80
  - tau = 10
  - gamma = 0.25
- Timing configuration: one environment, with per-episode warm-up
  policy queries removed before aggregation.

## Task performance

| Metric | Original | AASF | Paired difference | Statistical result |
|---|---:|---:|---:|---|
| Mean score | 0.93197 ± 0.21801 | 0.92459 ± 0.21804 | −0.00738 | Bootstrap 95% CI [−0.03672, 0.01985] |
| Score comparison | — | — | Win/tie/loss = 34/33/33 | Wilcoxon p = 0.7171 |
| Success rate, score ≥ 0.90 | 92/100 (92%) | 89/100 (89%) | −3 percentage points | 95% CI [−8, 1]; McNemar p = 0.375 |

No statistically significant task-performance difference was detected.
Formal equivalence or non-inferiority was not established.

## Action smoothness

Positive reductions indicate smoother AASF action sequences.

| Metric | Original | AASF | Relative reduction | Bootstrap 95% CI | Holm-adjusted p | Cohen dz |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 9.44204 ± 1.91270 | 8.37636 ± 1.49483 | 11.29% | [8.46%, 13.97%] | 9.11e-10 | 0.725 |
| D2 | 5.34846 ± 1.06841 | 4.46039 ± 0.80696 | 16.60% | [13.65%, 19.41%] | 2.51e-12 | 0.989 |
| D3 | 7.54541 ± 1.75228 | 6.52998 ± 1.47608 | 13.46% | [10.03%, 16.67%] | 2.49e-09 | 0.723 |

AASF was smoother in 82, 84, and 77 of the 100 paired
episodes for D1, D2, and D3, respectively.

## Computational overhead

- Net AASF post-processing overhead:
  0.371933 ± 0.009013 ms per policy query.
- Bootstrap 95% CI:
  [0.370162, 0.373677] ms.
- Relative to Original policy-inference time:
  0.062947%.
- Relative bootstrap 95% CI:
  [0.062611%, 0.063276%].

Differences between separately measured policy-inference or total-loop
times must not be interpreted as acceleration caused by AASF.

## Adaptive behavior

A total of 20,979 valid adaptive decisions were analyzed.

| Quantity | Result |
|---|---:|
| Median d_i | 8.100452 |
| Q1 / Q3 | 4.126363 / 18.695298 |
| IQR | 14.568935 |
| 90th percentile | 34.737932 |
| 95th percentile | 41.948274 |
| Fraction d_i > tau | 41.913342% |
| Mean alpha_i | 0.700889 ± 0.143199 |
| Fraction alpha_i = alpha_min | 19.333619% |
| Fraction alpha_i = alpha_max | 58.086658% |
| Fraction alpha_i < alpha_max | 41.913342% |

The equality between the fraction with d_i > tau and the fraction with
alpha_i < alpha_max confirms that the adaptive branch was activated as
designed.

## Representative episode

Seed 4300005 achieved a score of 1.0 for both methods.

- Original D3: 6.08080082
- AASF D3: 4.99363296
- Absolute D3 reduction: 1.08716787
- Relative D3 reduction: approximately 17.88%

## Supported paper conclusion

AASF significantly improves first-, second-, and third-order action
smoothness with negligible post-processing overhead, while no
statistically significant change in task performance was detected
across the 100 paired evaluation episodes.

## Scope limitations

- Results are based on one task and one trained policy checkpoint.
- The 100 seeds are evaluation episodes, not independent training runs.
- No physical-robot experiment has yet been conducted.
- Non-inferiority was not formally established.
