"""MLflow Run Context Provider for PRML manifests.

When an MLflow run starts in a working tree containing a `.prml.yaml` or
`prml.yaml` file (in CWD or any ancestor directory), this provider attaches a
small set of `prml.*` tags to the run. The most important tag is
`prml.manifest_hash`, the SHA-256 of the canonical bytes of the manifest as
defined in PRML v0.1 §3.

The provider is intentionally defensive: a malformed or partial manifest must
never break an MLflow run. All field extraction is best-effort.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from mlflow.tracking.context.abstract_context import RunContextProvider
except Exception:  # pragma: no cover - import guard only
    # Fallback shim so the module imports even without mlflow installed
    # (e.g. for static analysis). The plugin is useless without mlflow at
    # runtime, but importing this module must not crash.
    class RunContextProvider:  # type: ignore[no-redef]
        def in_context(self) -> bool:
            return False

        def tags(self) -> Dict[str, str]:
            return {}


from mlflow_falsify._canonical import manifest_hash

MANIFEST_FILENAMES = (".prml.yaml", "prml.yaml")


def _find_manifest(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from `start` (default CWD) looking for a PRML manifest."""
    try:
        current = (start or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return None
    for directory in (current, *current.parents):
        for name in MANIFEST_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        spec = yaml.safe_load(text)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    if not isinstance(spec, dict):
        return None
    return spec


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _relative_path(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except (OSError, ValueError):
        return str(path)


class FalsifyRunContextProvider(RunContextProvider):
    """Attach PRML manifest tags to every MLflow run started near a manifest."""

    def in_context(self) -> bool:
        return _find_manifest() is not None

    def tags(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        manifest_path = _find_manifest()
        if manifest_path is None:
            return result

        result["prml.manifest_path"] = _relative_path(manifest_path)

        spec = _load_manifest(manifest_path)
        if spec is None:
            return result

        # Hash the canonical bytes. Anything raised here is swallowed —
        # tag emission must never break the run.
        try:
            result["prml.manifest_hash"] = manifest_hash(spec)
        except Exception:
            pass

        version = _safe_str(spec.get("version"))
        if version is not None:
            result["prml.version"] = version

        metric = _safe_str(spec.get("metric"))
        if metric is not None:
            result["prml.metric"] = metric

        comparator = _safe_str(spec.get("comparator"))
        if comparator is not None:
            result["prml.comparator"] = comparator

        threshold = spec.get("threshold")
        threshold_str = _safe_str(threshold)
        if threshold_str is not None:
            result["prml.threshold"] = threshold_str

        dataset = spec.get("dataset")
        if isinstance(dataset, dict):
            dataset_id = _safe_str(dataset.get("id"))
            if dataset_id is not None:
                result["prml.dataset_id"] = dataset_id

        return result
