# Changelog

All notable changes to mlflow-falsify will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-06-19

### Fixed
- **Correctness (critical): manifest hash now matches the `falsify` reference for integer-valued v0.1 thresholds.** The vendored canonicalizer skipped the v0.1 integer→float coercion (`threshold: 1` must canonicalize as `1.0`) and used non-reference YAML emitter settings, so a claim locked through this plugin and verified with the `falsify` CLI produced a spurious `TAMPERED` verdict. The adapter now delegates to `falsify_prml.canonicalize`/`manifest_hash` (single source of truth) instead of a private copy.

### Changed
- Added `falsify>=0.3.8` as a dependency (the reference canonicalizer).
- Added reference-parity tests asserting the adapter hashes byte-identically to `falsify_prml` across float, integer, and small-float thresholds — the integer case the float-only fixture never caught.

## [0.2.1] - 2026-06-02

### Fixed
- Docs: corrected the EU AI Act Article 12 reference date to **2 December 2027** (the high-risk obligation date, deferred from 2 August 2026 by the EU Digital Omnibus). No code changes; republished so the PyPI README reflects the correct date.

## [0.2.0] - 2026-05-23

### Added
- `MLFLOW_FALSIFY_TAG_SCOPE` environment variable. When set to `experiment`, only `prml.manifest_hash` and `prml.manifest_path` attach per-run; the 5 descriptive tags (metric, comparator, threshold, dataset_id, version) are lifted off-run.
- `mlflow_falsify.tag_experiment(experiment=None, *, manifest_path=None)` helper. Sets the descriptive PRML tags at experiment level via `MlflowClient.set_experiment_tag`. Idempotent.
- README section: "HPO sweeps and tag scope".
- 4 new unit tests covering default scope, experiment scope, unknown-value fallback, and explicit-path computation.

### Why
Surfaced by [@smqd19](https://github.com/smqd19) on [mlflow/mlflow#23369](https://github.com/mlflow/mlflow/discussions/23369): at 50k HPO runs/day, emitting 5 identical descriptive tags per-run is wasteful. The audit-essential commitment (`prml.manifest_hash`) must stay per-run; the descriptive fields can live at experiment scope. Tracks [#1](https://github.com/studio-11-co/mlflow-falsify/issues/1).

### Backward compat
Default behaviour unchanged. Without setting the env var, all 7 tags attach per-run exactly as in v0.1.x.

## [0.1.3] - 2026-05-20
- README: switch DOI badge to shields.io format for reliable rendering.
- README: add Audit & compliance crosswalks section (EU AI Act Art. 12, NIST AI RMF, ISO/IEC 42001).

## [0.1.2] - 2026-05-16
- CI: defer `FalsifyRunContextProvider` import to break MLflow circular load during entry-point discovery.
- Add OpenSSF Scorecard workflow.

## [0.1.1] - 2026-05-15
- First public release on PyPI.
- 5 unittests including TV-001 conformance assertion against the PRML v0.1 normative vector.
- Discovery via standard `mlflow.run_context_provider` entry point.
- Canonicalize logic vendored byte-for-byte from `falsify` v0.1.4.
