# Reproducibility notes

The complete guide is [`docs/AASF_BSOP_REPRODUCIBILITY.md`](docs/AASF_BSOP_REPRODUCIBILITY.md).

## Included

- Frozen AASF, BSOP, SG, and Temporal Ensemble implementation files.
- Final condition-specific and seven-condition analysis scripts.
- Compact per-seed tables, statistical outputs, and frozen manifests.
- SG and TE development/confirmatory records.
- Final post-processing timing microbenchmark.
- Compact Franka simulated-replay results.
- SO-101 split, checkpoint configuration, 10 held-out boundary arrays, and
  offline audit results.

## Excluded

- Model weights and optimizer tensors.
- Source datasets, video files, and server caches.
- Bulk raw simulation traces and Franka time-series files.
- Failed/retry runs, PID files, and superseded backups.

Raw data should be restored in the paths documented by the full guide. Validate
the repository payload with:

```bash
sha256sum -c AASF_BSOP_V1.1.0_MANIFEST.sha256
```

Absence of a statistically significant score difference is not proof of
equivalence or non-inferiority. SO-101 results are teacher-forced offline
trajectory results, and Franka results are simulated dynamics replay.
