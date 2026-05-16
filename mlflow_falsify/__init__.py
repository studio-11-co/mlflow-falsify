"""mlflow-falsify — automatic PRML manifest hash tagging for MLflow runs.

Lazy import: importing ``FalsifyRunContextProvider`` at module top-level
causes a circular import when MLflow's entry-points discovery loads this
package (MLflow imports ``mlflow_falsify.run_context:FalsifyRunContextProvider``
during ``register_entrypoints()``, but ``run_context.py`` itself imports from
``mlflow.tracking.context.abstract_context`` — mlflow is mid-init at that
point). PEP 562 ``__getattr__`` defers the import to first access.
"""

from __future__ import annotations

__version__ = "0.1.3"
__all__ = ["FalsifyRunContextProvider"]


def __getattr__(name: str):
    if name == "FalsifyRunContextProvider":
        from mlflow_falsify.run_context import FalsifyRunContextProvider
        return FalsifyRunContextProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
