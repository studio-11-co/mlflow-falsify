# mlflow-falsify — automatic PRML manifest hash tagging for MLflow runs

[![PyPI version](https://img.shields.io/pypi/v/mlflow-falsify.svg)](https://pypi.org/project/mlflow-falsify/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlflow-falsify.svg)](https://pypi.org/project/mlflow-falsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20235451.svg)](https://doi.org/10.5281/zenodo.20235451)
[![Spec: PRML v0.1](https://img.shields.io/badge/spec-PRML%20v0.1-39d98a)](https://spec.falsify.dev/v0.1)

Drop a PRML manifest in your repo. Every MLflow run gets cryptographically bound to it. No workflow changes.

## Install

```bash
pip install mlflow-falsify
```

The plugin is discovered automatically through MLflow's `mlflow.run_context_provider` entry point.

## Usage

```python
import mlflow

# .prml.yaml exists in CWD or any parent directory — that's all you need.
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.873)
    # The run now carries prml.manifest_hash and friends as tags.
```

## What gets tagged

When a `.prml.yaml` or `prml.yaml` is found in the current directory or any ancestor, every run is tagged with:

- `prml.manifest_hash` — SHA-256 of the canonical manifest bytes (PRML v0.1 §3)
- `prml.manifest_path` — relative path to the discovered manifest
- `prml.version` — manifest schema version (e.g. `prml/0.1`)
- `prml.metric` — the pre-registered metric (e.g. `accuracy`)
- `prml.comparator` — one of `>=`, `>`, `==`, `<=`, `<`
- `prml.threshold` — the numeric threshold, as a string
- `prml.dataset_id` — the pre-registered dataset identifier

Missing or malformed fields are silently skipped. The provider never raises into your run.

## Why this matters

- **EU AI Act Article 12 evidence layer.** Every logged run carries a tamper-evident pointer to the claim it was meant to test.
- **Eval reproducibility by default.** The hash freezes metric, threshold, dataset, and seed before the experiment runs.
- **Audit trails for free.** Reviewers can recompute the manifest hash from the YAML and compare it against your tracked runs.
- **No workflow change.** Existing MLflow code is untouched — the plugin attaches via entry points.

## Links

- PRML specification: [spec.falsify.dev/v0.1](https://spec.falsify.dev/v0.1)
- Falsify reference implementation: [github.com/studio-11-co/falsify](https://github.com/studio-11-co/falsify)
- Zenodo (this plugin): [doi.org/10.5281/zenodo.20235451](https://doi.org/10.5281/zenodo.20235451)
- Zenodo (PRML spec): [doi.org/10.5281/zenodo.20177839](https://doi.org/10.5281/zenodo.20177839)
- Registry: [registry.falsify.dev](https://registry.falsify.dev)

## Audit & compliance crosswalks

Where the manifest hash this plugin attaches fits in major AI governance frameworks (FULL / PARTIAL / NONE tagged):

- [EU AI Act Article 12](https://spec.falsify.dev/eu-ai-act/article-12/) — automated-logging pattern for the 2 August 2026 high-risk deadline
- [NIST AI RMF 1.0](https://spec.falsify.dev/nist-ai-rmf/) — GOVERN / MAP / MEASURE / MANAGE subcategory map
- [ISO/IEC 42001:2023](https://spec.falsify.dev/iso-42001/) — AI Management System clause-by-clause evidence map

## License

MIT. Copyright 2026 Studio 11 / Cüneyt Öztürk.
