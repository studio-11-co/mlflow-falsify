"""mlflow-falsify — automatic PRML manifest hash tagging for MLflow runs.

Lazy import: importing ``FalsifyRunContextProvider`` at module top-level
causes a circular import when MLflow's entry-points discovery loads this
package (MLflow imports ``mlflow_falsify.run_context:FalsifyRunContextProvider``
during ``register_entrypoints()``, but ``run_context.py`` itself imports from
``mlflow.tracking.context.abstract_context`` — mlflow is mid-init at that
point). PEP 562 ``__getattr__`` defers the import to first access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

# Single-sourced from package metadata (pyproject.toml) so the self-reported
# version can never drift from the released wheel again (0.2.2 shipped
# reporting "0.2.1"). The fallback covers running from a checkout where the
# package is not installed.
try:
    from importlib.metadata import version as _v
    __version__ = _v("mlflow-falsify")
except Exception:
    __version__ = "0.2.3"
__all__ = ["FalsifyRunContextProvider", "tag_experiment"]


def __getattr__(name: str):
    if name == "FalsifyRunContextProvider":
        from mlflow_falsify.run_context import FalsifyRunContextProvider
        return FalsifyRunContextProvider
    if name == "tag_experiment":
        return tag_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tag_experiment(
    experiment: Optional[Union[str, int]] = None,
    *,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Lift the descriptive PRML tags to experiment-level once per sweep.

    For HPO sweeps where a single PRML claim is reused across thousands of
    runs, repeating ``prml.metric`` / ``prml.comparator`` / ``prml.threshold``
    / ``prml.dataset_id`` / ``prml.version`` per-run is wasteful. Setting
    ``MLFLOW_FALSIFY_TAG_SCOPE=experiment`` keeps only ``prml.manifest_hash``
    and ``prml.manifest_path`` at the run level; this helper sets the rest
    once at experiment level via ``MlflowClient.set_experiment_tag``.

    Idempotent: calling twice with the same manifest is a no-op.

    Parameters
    ----------
    experiment :
        Experiment name (``str``) or experiment id (``int`` or numeric
        ``str``). If omitted, the currently active MLflow experiment is
        used.
    manifest_path :
        Optional explicit path to a ``.prml.yaml`` manifest. If omitted,
        the manifest is auto-discovered from CWD upwards (the same rule
        the provider uses).

    Returns
    -------
    Dict[str, str]
        The tags that were written. Empty dict if no manifest was found
        or the manifest contained no descriptive fields.

    Raises
    ------
    RuntimeError
        If ``mlflow`` is not installed.
    ValueError
        If the named experiment cannot be resolved.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mlflow_falsify.tag_experiment requires mlflow to be installed"
        ) from exc

    from mlflow_falsify.run_context import _RUN_LEVEL_ALWAYS, _compute_tags

    path = Path(manifest_path) if manifest_path is not None else None
    all_tags = _compute_tags(path)
    descriptive = {k: v for k, v in all_tags.items() if k not in _RUN_LEVEL_ALWAYS}
    if not descriptive:
        return {}

    client = MlflowClient()

    if experiment is None:
        exp_id = mlflow.tracking.fluent._get_experiment_id()
    elif isinstance(experiment, int) or (
        isinstance(experiment, str) and experiment.isdigit()
    ):
        exp_id = str(experiment)
    else:
        exp = client.get_experiment_by_name(str(experiment))
        if exp is None:
            raise ValueError(f"experiment {experiment!r} not found")
        exp_id = exp.experiment_id

    for key, value in descriptive.items():
        client.set_experiment_tag(exp_id, key, value)

    return descriptive
