# BSOP+AASF Fresh-50 Confirmatory Analysis Record

## Frozen experiment

- Frozen commit: 01ee28858cc3cc8a46438351f559cc1b8221afac
- Frozen tag: bsop-aasf-confirmatory-frozen-20260805
- Seeds: 4300100–4300149
- Episodes per method: 50
- Methods: Original, AASF, BSOP, BSOP+AASF
- Protocol SHA256:
  be0caf893b25ac8da7f8a476ab1833aec6170e830a30c6764d44c1c3bc5be2dd
- Checkpoint SHA256:
  f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876
- Original artifact-manifest SHA256:
  f49d3afd546fdb7bbcc6d1deb600a6615ee026f1e22d7dbc85b69870741d6914

## Locked BSOP+AASF parameters

- eta = 1.0
- lambda_max = 0.5
- rho = 5.0
- kappa = 0.5

## Same-boundary mechanism result

- Online transitions: 1,504
- BSOP activation rate: 69.08%
- Raw boundary-jump mean: 12.671189
- Projected boundary-jump mean: 10.828972
- Pooled reduction: 14.5386%
- Mean per-seed reduction: 14.5563%
- 95% bootstrap CI: [12.9596%, 16.3118%]
- Improved episodes: 50/50
- Boundary / absolute / relative violations: 0 / 0 / 0

## Primary comparison: AASF to BSOP+AASF

- Mean score difference: -0.0077
- 95% bootstrap CI: [-0.0222, +0.0025]
- SR@0.90: 46/50 to 45/50
- Exact success: 31/50 to 26/50
- Exact-success paired risk difference: -10 percentage points
- Exact-success bootstrap 95% CI: [-22, +2] percentage points
- Exact-success McNemar p: 0.226562
- Common-prefix D1 reduction: 0.1133%
- Common-prefix D2 reduction: 8.2545%
- Common-prefix D3 reduction: 14.7799%
- Own-trajectory boundary-D1 reduction: 15.0884%

## Primary comparison: Original to BSOP+AASF

- Mean score difference: -0.0150
- 95% bootstrap CI: [-0.0504, +0.0111]
- SR@0.90: 46/50 to 45/50
- Exact success: 27/50 to 26/50
- Exact-success paired risk difference: -2 percentage points
- Exact-success bootstrap 95% CI: [-18, +14] percentage points
- Exact-success McNemar p: 1
- Common-prefix D1 reduction: 7.3937%
- Common-prefix D2 reduction: 22.9816%
- Common-prefix D3 reduction: 26.8877%
- Own-trajectory boundary-D1 reduction: 10.6493%

## Interpretation lock

The confirmatory results support deterministic boundary-safety
and substantial reductions in boundary discontinuity and higher-order
action differences.

They do not establish task-performance non-inferiority. Relative to
AASF, the exact-success result shows an unfavorable but statistically
uncertain direction. The method must therefore be presented as a
quantified smoothness–task-performance trade-off, not as uniformly
superior or performance-preserving without qualification.

No method parameters are to be changed in response to these results.
