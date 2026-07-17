"""Verify a run's logged metric against the locked PRML predicate.

The run-context provider (run_context.py) binds a run to a manifest hash at
start. This module closes the loop at run end: it reads the metric named in
the manifest from the run's logged metrics, evaluates the locked
comparator/threshold, and writes the outcome back as tags:

    prml.verdict   PASS | FAIL | UNVERIFIED | TAMPERED
    prml.observed  the metric value the verdict was computed from

TAMPERED means the manifest file changed between run start and verification —
the recomputed canonical hash no longer matches the hash the run was tagged
with at start. UNVERIFIED means the manifest's metric was never logged.

Same defensive contract as the provider: verification must never break an
MLflow run. `locked_run()` re-raises nothing of its own; failures degrade to
an UNVERIFIED tag. Set MLFLOW_FALSIFY_STRICT=1 to raise FalsifyVerdictError
on FAIL or TAMPERED instead (for CI use).
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from mlflow_falsify._canonical import manifest_hash
from mlflow_falsify.run_context import _find_manifest, _load_manifest

# Spec §5.1: `==` compares within a tolerance (default 1e-9, overridable via
# metric_args.tolerance). Mirrors the falsify 0.3.11 reference behaviour.
_DEFAULT_TOLERANCE = 1e-9


class FalsifyVerdictError(RuntimeError):
    """Raised in strict mode when the verdict is FAIL or TAMPERED."""


def _strict() -> bool:
    return os.environ.get("MLFLOW_FALSIFY_STRICT", "").strip() in ("1", "true", "yes")


def evaluate_predicate(observed: float, comparator: str, threshold: float,
                       tolerance: float = _DEFAULT_TOLERANCE) -> bool:
    if comparator == ">=":
        return observed >= threshold
    if comparator == "<=":
        return observed <= threshold
    if comparator == ">":
        return observed > threshold
    if comparator == "<":
        return observed < threshold
    if comparator == "==":
        return abs(observed - threshold) < tolerance
    raise ValueError(f"invalid comparator: {comparator}")


def _tolerance_from(spec: Dict[str, Any]) -> float:
    args = spec.get("metric_args")
    if isinstance(args, dict):
        tol = args.get("tolerance")
        if isinstance(tol, (int, float)) and not isinstance(tol, bool):
            return float(tol)
    return _DEFAULT_TOLERANCE


def verify_run(run_id: Optional[str] = None,
               manifest_path: Optional[Path] = None,
               expected_hash: Optional[str] = None) -> str:
    """Verify `run_id` (default: the active run) against the PRML manifest.

    Returns the verdict string and tags the run. Never raises unless
    MLFLOW_FALSIFY_STRICT=1 and the verdict is FAIL or TAMPERED.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if run_id is None:
        active = mlflow.active_run()
        if active is None:
            warnings.warn("mlflow-falsify: no active run to verify", stacklevel=2)
            return "UNVERIFIED"
        run_id = active.info.run_id

    def _tag(verdict: str, observed: Optional[float] = None) -> str:
        try:
            client.set_tag(run_id, "prml.verdict", verdict)
            if observed is not None:
                client.set_tag(run_id, "prml.observed", str(observed))
        except Exception:
            pass
        if verdict in ("FAIL", "TAMPERED") and _strict():
            raise FalsifyVerdictError(f"prml.verdict={verdict} for run {run_id}")
        return verdict

    if manifest_path is None:
        manifest_path = _find_manifest()
    if manifest_path is None:
        return _tag("UNVERIFIED")
    spec = _load_manifest(manifest_path)
    if spec is None:
        return _tag("UNVERIFIED")

    run = client.get_run(run_id)

    # Tamper check: does the file still hash to what the run was bound to?
    # Prefer the tag the context provider set at run start; fall back to the
    # hash locked_run() captured at entry (provider may be absent).
    bound = run.data.tags.get("prml.manifest_hash") or expected_hash
    if bound:
        try:
            current = manifest_hash(spec)
        except Exception:
            current = None
        if current is not None and current != bound:
            return _tag("TAMPERED")

    metric_name = spec.get("metric")
    comparator = spec.get("comparator")
    threshold = spec.get("threshold")
    if (not isinstance(metric_name, str)
            or comparator not in (">=", "<=", ">", "<", "==")
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))):
        return _tag("UNVERIFIED")

    observed = run.data.metrics.get(metric_name)
    if observed is None:
        return _tag("UNVERIFIED")

    ok = evaluate_predicate(float(observed), comparator, float(threshold),
                            _tolerance_from(spec))
    return _tag("PASS" if ok else "FAIL", float(observed))


@contextmanager
def locked_run(**start_run_kwargs: Any) -> Iterator[Any]:
    """`mlflow.start_run()` that verifies the locked predicate on exit.

    Usage:
        with mlflow_falsify.locked_run():
            mlflow.log_metric("accuracy", 0.91)
        # run is now tagged prml.verdict=PASS/FAIL/... automatically

    On an exception inside the block the run is left unverdicted (MLflow marks
    it FAILED anyway); the manifest tamper check still runs.
    """
    import mlflow

    manifest_path = _find_manifest()
    entry_hash: Optional[str] = None
    if manifest_path is not None:
        spec = _load_manifest(manifest_path)
        if spec is not None:
            try:
                entry_hash = manifest_hash(spec)
            except Exception:
                entry_hash = None
    with mlflow.start_run(**start_run_kwargs) as run:
        try:
            yield run
        finally:
            try:
                verify_run(run.info.run_id, manifest_path=manifest_path,
                           expected_hash=entry_hash)
            except FalsifyVerdictError:
                raise
            except Exception:
                warnings.warn("mlflow-falsify: verification failed softly",
                              stacklevel=2)
