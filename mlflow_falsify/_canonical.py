"""Canonical YAML serialization for PRML manifests.

Vendored byte-for-byte from falsify v0.1.4 (`falsify.py::_canonicalize`).
Reference: Zenodo DOI 10.5281/zenodo.20177839. Spec: https://spec.falsify.dev/v0.1
"""

from __future__ import annotations

import hashlib
from typing import Any

import yaml


def canonicalize(spec: Any) -> str:
    """Render a YAML tree in a stable form suitable for hashing."""
    return yaml.safe_dump(
        spec,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )


def manifest_hash(spec: Any) -> str:
    """Return the lowercase hex SHA-256 of the canonical bytes of `spec`."""
    return hashlib.sha256(canonicalize(spec).encode("utf-8")).hexdigest()
