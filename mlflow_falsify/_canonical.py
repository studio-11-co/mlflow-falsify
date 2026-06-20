"""Canonical serialization for PRML manifests.

Delegates to the reference implementation in ``falsify_prml`` (the ``falsify``
package) so this adapter is guaranteed byte-identical to the spec and the
other reference impls (py/js/go/rust).

Do NOT vendor a private copy of the canonicalizer here: any drift (e.g. a
missing v0.1 integer->float coercion, or different YAML emitter settings)
produces a different SHA-256 and therefore a spurious TAMPERED verdict — the
exact failure mode falsify exists to prevent.

The ``falsify_prml`` import is deliberately lazy (inside the functions, not at
module top level): this module is imported transitively while MLflow is still
registering its run-context-provider entry points, and pulling another package
into that window can trip MLflow's partially-initialised-module guard. By the
time these functions are actually called (run tag time), MLflow has finished
importing.

Spec: https://spec.falsify.dev/v0.1
"""

from __future__ import annotations

from typing import Any

__all__ = ["canonicalize", "manifest_hash"]


def canonicalize(spec: Any) -> str:
    """Render a PRML manifest in canonical form (reference implementation)."""
    from falsify_prml import canonicalize as _canonicalize

    return _canonicalize(spec)


def manifest_hash(spec: Any) -> str:
    """Return the lowercase hex SHA-256 of the canonical bytes of `spec`."""
    from falsify_prml import manifest_hash as _manifest_hash

    return _manifest_hash(spec)
