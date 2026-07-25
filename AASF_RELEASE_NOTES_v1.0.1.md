# AASF-Diffusion-Policy v1.0.1 Release Notes

This release completes the AASF CRN reproducibility package.

It adds:

- frozen Original and AASF results for 100 paired evaluation seeds;
- final task-performance, smoothness, timing, and sensitivity analyses;
- manuscript-ready figures and tables;
- the three previously omitted runtime and reproducibility files;
- a portable baseline launcher whose project root is derived from its own location;
- SHA-256 manifests for the repository payload.

Safety and attribution:

- The installer does not replace the repository-root `README.md`, `LICENSE`, or
  `CITATION.cff`.
- The upstream Diffusion Policy README and license are retained under
  `docs/upstream_diffusion_policy/`.
- Existing payload files are backed up before replacement.
- Installation does not commit or push changes.

Known exclusions:

- The trained checkpoint is not included.
- The 200 raw per-episode NPZ traces are not included.
- Their expected locations and recorded SHA-256 values remain documented in the
  reproducibility materials.
