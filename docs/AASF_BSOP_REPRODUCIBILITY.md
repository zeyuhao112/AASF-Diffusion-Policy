# AASF–BSOP reproducibility guide

## Release scope

This repository records the frozen implementation, analysis code, compact
tables, and provenance for the AASF–BSOP manuscript. The public Git tree is a
source-and-evidence package; it is not a mirror of either experiment server.

The release intentionally omits model weights, datasets, video caches, failed
runs, backup scripts, PID files, and bulk raw simulation traces.

## Environment

The upstream Diffusion Policy environment remains defined by
`conda_environment.yaml`. Run analysis commands from the repository root.
Several statistical scripts additionally require NumPy, pandas, and SciPy.

The frozen results were produced on Linux. Simulation command metrics use an
executed control stream at 10 Hz (`dt=0.1 s`). C1, C2, and C3 are
time-normalized finite-difference command proxies; they are not direct physical
measurements of link acceleration, torque, or mechanical jerk.

## Seven-condition BSOP analysis

The canonical analysis is:

```bash
python paper_exp/scripts/analyze_bsop_time_normalized_7cond.py
```

It expects these trace directories under `paper_artifacts/traces/`:

- `bsop_aasf_confirmatory_fresh50`
- `bsop_aasf_multicheckpoint_seed43_fresh50`
- `bsop_aasf_multicheckpoint_seed44_fresh50`
- `blockpush_bsop_aasf_confirmatory_fresh50`
- `pusht_image_bsop_confirmatory_fresh50`
- `pusht_image_bsop_multicheckpoint_seed43_fresh50`
- `pusht_image_bsop_multicheckpoint_seed44_fresh50`

The raw traces are excluded from Git. Their frozen hashes are retained under
`paper_exp/metadata/<run>/`. With the raw traces restored, the canonical script
regenerates the 12 files in
`paper_exp/tables/bsop_time_normalized_7cond_v1/` and must end with:

```text
BSOP_TIME_NORMALIZED_7COND = PASS
```

The archived release was independently checked from all 350 paired formal
seeds; all 12 regenerated outputs matched the frozen files byte for byte.

## Four-method analyses

The low-dimensional Push-T checkpoints use:

```bash
python paper_exp/analysis/analyze_bsop_aasf_confirmatory_fresh50.py
python paper_exp/analysis/analyze_multicheckpoint_seed43_fresh50.py
python paper_exp/analysis/analyze_multicheckpoint_seed44_fresh50.py
```

BlockPush uses:

```bash
python paper_exp/analysis/analyze_blockpush_confirmatory_fresh50.py
```

The four BlockPush `eval_log.json` files are included under
`paper_exp/results/blockpush_bsop_aasf_confirmatory_fresh50/`; its 200 NPZ
execution traces remain external.

## AASF, SG, and Temporal Ensemble baselines

- Three-checkpoint image AASF analysis:
  `paper_exp/scripts/analyze_pusht_image_aasf_multicheckpoint_3cond_20260818.py`
- Savitzky–Golay selection and confirmatory record:
  `paper_exp/baselines/savgol_seed42/`
- Fresh Temporal Ensemble selection and confirmatory record:
  `paper_exp/baselines/temporal_ensemble_seed42/`
- Same-seed TE comparison:
  `paper_exp/baselines/temporal_ensemble_same_seed_seed42/`
- Combined post-processing microbenchmark:
  `paper_exp/timing/bsop_te_aasf_combined/`

The SG and TE development seeds are disjoint from their formal confirmatory
seeds. Superseded and failed attempts are intentionally not published.

## Franka simulated dynamics replay

The frozen protocol and compact per-seed results are under
`paper_artifacts/franka_replay/`. Raw per-trajectory time-series files are not
included. This experiment maps the recorded command sequences into a MuJoCo
Franka Panda replay and measures simulated tracking, torque-rate, and
end-effector jerk quantities. It is not a real-hardware vibration experiment.

## SO-101 held-out offline audit

See `paper_artifacts/so101_heldout10/README.md`. The included compact package
contains the split manifest, final checkpoint configuration (without weights),
10 held-out boundary arrays, projection outputs, and policy-quality summaries.

The primary BSOP reference is the recorded action immediately preceding each
query. The analysis is teacher-forced and offline: it does not establish
closed-loop success, safety, or hardware vibration reduction.

## Integrity checks

The release-level manifest is `AASF_BSOP_V1.1.0_MANIFEST.sha256`. Verify it
from the repository root with:

```bash
sha256sum -c AASF_BSOP_V1.1.0_MANIFEST.sha256
```

The manifest excludes itself and Git metadata. Historical manifests are kept
only when their referenced payload is present and self-contained.
