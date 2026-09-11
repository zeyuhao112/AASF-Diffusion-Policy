# Experiment artifacts

This directory contains compact final experiment records selected from the
frozen 3090 study workspace.

- `analysis/`: per-condition four-method analysis.
- `scripts/`: canonical cross-condition, image, SG, and TE scripts and launchers.
- `tables/`: manuscript-facing per-seed tables, summaries, cross-checks, and
  multiplicity corrections.
- `metadata/`: frozen raw-input hashes and run metadata.
- `baselines/`: compact SG and TE development/confirmatory records.
- `ablations/`: Table-10 AASF parameter-selection sources.
- `timing/`: final combined post-processing microbenchmark.
- `results/`: small source logs required by standalone analyses.

Raw simulation NPZ traces are excluded from Git. Restore them under
`paper_artifacts/traces/` before running analyses that load trajectories. See
`docs/AASF_BSOP_REPRODUCIBILITY.md` for the required directory names.
