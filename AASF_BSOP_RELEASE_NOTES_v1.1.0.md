# AASF–BSOP Diffusion Policy v1.1.0

This release extends the AASF reproducibility package with the manuscript's
frozen BSOP, baseline, cross-condition, and offline robot-trajectory evidence.

## Added

- BSOP evaluation support for Push-T low-dimensional, Push-T image, and
  BlockPush low-dimensional runners.
- Absolute-time-aligned Temporal Ensemble support and formal SG/TE baselines.
- Seven-condition BSOP analysis with 350 paired formal seeds, time-normalized
  C1–C3 command metrics, canonical cross-checks, and multiplicity correction.
- Three-checkpoint zero-retuning image AASF replication.
- Final combined AASF, BSOP, BSOP+AASF, and TE timing microbenchmark.
- Frozen MuJoCo Franka dynamics replay outputs.
- Compact SO-101 held-out teacher-forced trajectory audit without model weights.
- Table-10 AASF threshold and gain parameter-selection records.

## Validation

- All seven-condition raw trace hashes were checked.
- The canonical seven-condition analysis regenerated all 12 frozen outputs byte
  for byte and ended with `BSOP_TIME_NORMALIZED_7COND = PASS`.
- Standalone low-dimensional and BlockPush four-method analyses regenerated
  their frozen outputs byte for byte.
- SO-101 split, boundary-export, projection, and policy-quality artifacts were
  checked against their frozen manifests.

## Excluded

- Training checkpoints and model weights.
- Dataset and Hugging Face caches.
- Bulk raw simulation and Franka time-series traces.
- Videos, failed attempts, restart controllers, PID files, and superseded
  backup scripts.

These larger artifacts should be distributed separately as versioned release
assets or an institutional data supplement.
