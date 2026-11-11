# AASF Table-10 parameter selection

This compact record contains the threshold and gain tables, their 50-seed source
logs, and the scale-aware threshold regression. Videos and duplicate copies are
excluded.

Run from the repository root:

```bash
python paper_exp/ablations/aasf_table10/verify_table10.py
```

The verifier recomputes mean score, SR@0.90, SR@0.95, failure count, D1, D2,
and D3 from seeds 4300000–4300049 and compares them with both CSV tables.

The confirmatory seven-condition experiments use disjoint seeds
4300100–4300149.
