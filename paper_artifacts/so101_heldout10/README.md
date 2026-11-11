# SO-101 held-out trajectory audit

This directory contains the compact, frozen offline audit used for the SO-101
section of the manuscript.

## Protocol

- Source dataset: `lerobot/svla_so101_pickplace`
- Split: 40 training episodes and 10 held-out episodes
- Held-out original indices: 1, 3, 8, 11, 14, 19, 22, 38, 39, 47
- Training: Diffusion Policy, seed 1000, 20,000 steps, batch size 4
- Action dimension: 6
- Horizon / observations / executed chunk: 16 / 2 / 8
- Recorded frequency: 30 Hz
- Held-out export: 273 queries and 263 adjacent query boundaries

The final model weights, source videos, and dataset cache are not included.
`checkpoint_config/` contains configuration and state metadata only.

## Directory map

- `scripts/so101_make_40_10_split.py`: deterministic dataset split.
- `scripts/so101_heldout10_boundary_export.py`: frozen held-out prediction
  and boundary export.
- `scripts/so101_heldout10_bsop_projection_audit.py`: offline BSOP projection
  and episode-cluster bootstrap.
- `scripts/so101_heldout_policy_quality.py`: prediction-versus-recorded-action
  alignment audit.
- `boundary_export/`: 10 compact episode NPZ files, boundary CSV, and summary.
- `bsop_projection/`: primary and supplementary projection rows and summary.
- `policy_quality/`: per-query and per-episode prediction-quality records.

## Primary result

Using the recorded action as the primary boundary reference, BSOP changes the
mean normalized boundary discrepancy from 0.190532388 to 0.163906747, a pooled
reduction of 13.974339%. The episode-cluster 95% bootstrap interval is
[11.504938%, 16.098201%]. BSOP is active on 167 of 263 boundaries, with no
non-increase, absolute-cap, or relative-cap violations.

The mean applied interpolation coefficient is 0.160965 across all 263
boundaries and 0.253496 among the 167 active boundaries.

## Interpretation limit

This is a teacher-forced offline analysis of trajectories recorded on a real
SO-101 platform. It is not online deployment of BSOP, does not measure task
success under closed-loop BSOP control, and is not physical vibration or
hardware-safety validation.
