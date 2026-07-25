# Reproducibility Notes

## Included in this core bundle

- Final AASF and Original implementation files.
- Evaluation runner and baseline execution script.
- Environment specification.
- Original and AASF final evaluation logs.
- All final statistical-analysis scripts.
- Final CSV, JSON, text, and console reports.
- Manuscript-ready PNG and PDF figures.
- SHA-256 manifests.

## Raw trace policy

The 200 per-episode NPZ traces are intentionally excluded from this
lightweight core archive. Their paths and SHA-256 values are recorded in:

`paper_exp/tables/crn_final_trace_inputs.sha256`

The expected source directories are:

- `paper_artifacts/traces/crn_final_original_100`
- `paper_artifacts/traces/crn_final_aasf_100`

The raw traces should be archived separately or uploaded as a repository
release / institutional-data supplement.

## Analysis commands

Run commands from the repository root.

Task-performance analysis:

```bash
python paper_exp/scripts/crn_final_paired_analysis.py
```

Timing, adaptive-coefficient, and representative-episode analysis:

```bash
python paper_exp/scripts/crn_final_trace_analysis.py
```

Paired D1, D2, and D3 analysis:

```bash
python paper_exp/scripts/crn_final_smoothness_paired_analysis.py
```

## Statistical interpretation

- Smoothness reductions are defined as Original minus AASF, so positive
  values indicate improvement.
- Holm correction is applied across D1, D2, and D3.
- The score analysis did not detect a statistically significant
  difference.
- Absence of significance must not be described as proof of equivalence
  or non-inferiority.
- The measured AASF contribution is its net post-processing overhead;
  separate policy-inference timing differences are not algorithmic
  acceleration.
