"""Tests for the Falsify MLflow Run Context Provider.

Run with: python -m unittest discover -s tests

These tests do not require mlflow to be installed. The provider's parent class
is stubbed out in `mlflow_falsify.run_context` when mlflow is unavailable; the
tag-extraction logic is exercised directly.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from mlflow_falsify.run_context import FalsifyRunContextProvider

# TV-001 from spec/test-vectors/v0.1/test-vectors.json — the minimal valid
# manifest. Its canonical hash is fixed by the PRML v0.1 conformance suite.
TV_001_YAML = textwrap.dedent(
    """\
    version: prml/0.1
    claim_id: 01900000-0000-7000-8000-000000000000
    created_at: '2026-05-01T12:00:00Z'
    metric: accuracy
    comparator: '>='
    threshold: 0.85
    dataset:
      id: imagenet-val-2012
      hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    seed: 42
    producer:
      id: studio-11.co
    """
)

TV_001_HASH = "1a3466cc08ee7fb60a726ea1c4db6ecf48a9f847b9b7523bfb54b2ffaefee546"


class _Chdir:
    """Context manager: change CWD for the duration of a `with` block."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self._previous: str = ""

    def __enter__(self) -> None:
        self._previous = os.getcwd()
        os.chdir(self.target)

    def __exit__(self, *exc: object) -> None:
        os.chdir(self._previous)


class RunContextTests(unittest.TestCase):
    def test_in_context_returns_true_when_manifest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(TV_001_YAML, encoding="utf-8")
            with _Chdir(root):
                provider = FalsifyRunContextProvider()
                self.assertTrue(provider.in_context())

    def test_in_context_returns_false_when_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with _Chdir(Path(tmp).resolve()):
                provider = FalsifyRunContextProvider()
                self.assertFalse(provider.in_context())

    def test_tags_extracts_canonical_hash_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = root / ".prml.yaml"
            manifest.write_text(TV_001_YAML, encoding="utf-8")
            with _Chdir(root):
                provider = FalsifyRunContextProvider()
                tags = provider.tags()

        self.assertEqual(tags.get("prml.manifest_hash"), TV_001_HASH)
        self.assertEqual(tags.get("prml.version"), "prml/0.1")
        self.assertEqual(tags.get("prml.metric"), "accuracy")
        self.assertEqual(tags.get("prml.comparator"), ">=")
        self.assertEqual(tags.get("prml.threshold"), "0.85")
        self.assertEqual(tags.get("prml.dataset_id"), "imagenet-val-2012")
        self.assertIn("prml.manifest_path", tags)

    def test_manifest_found_in_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(TV_001_YAML, encoding="utf-8")
            nested = root / "experiments" / "run-001"
            nested.mkdir(parents=True)
            with _Chdir(nested):
                provider = FalsifyRunContextProvider()
                self.assertTrue(provider.in_context())
                tags = provider.tags()
                self.assertEqual(tags.get("prml.manifest_hash"), TV_001_HASH)

    def test_malformed_manifest_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(": not valid yaml :\n\t- [", encoding="utf-8")
            with _Chdir(root):
                provider = FalsifyRunContextProvider()
                # in_context still returns True — the file exists.
                self.assertTrue(provider.in_context())
                tags = provider.tags()
                # No hash because parsing failed, but path is recoverable.
                self.assertIn("prml.manifest_path", tags)
                self.assertNotIn("prml.manifest_hash", tags)


class ReferenceParityTests(unittest.TestCase):
    """The adapter MUST hash byte-identically to the falsify_prml reference.

    Any divergence produces a spurious TAMPERED verdict when a claim locked via
    this plugin is verified with the `falsify` CLI. The earlier vendored
    canonicalizer diverged on integer-valued v0.1 thresholds (it skipped the
    int->float coercion), which the float-only TV-001 fixture never caught.
    """

    def _parity(self, manifest: dict) -> None:
        import falsify_prml
        from mlflow_falsify._canonical import manifest_hash

        self.assertEqual(
            manifest_hash(manifest),
            falsify_prml.manifest_hash(manifest),
            f"adapter hash diverged from falsify_prml reference for {manifest!r}",
        )

    def _base(self, threshold) -> dict:
        return {
            "version": "prml/0.1",
            "claim_id": "01900000-0000-7000-8000-000000000000",
            "created_at": "2026-05-01T12:00:00Z",
            "metric": "accuracy",
            "comparator": ">=",
            "threshold": threshold,
            "dataset": {
                "id": "imagenet-val-2012",
                "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            "seed": 42,
            "producer": {"id": "studio-11.co"},
        }

    def test_float_threshold_parity(self) -> None:
        self._parity(self._base(0.85))

    def test_integer_threshold_parity(self) -> None:
        # The regression case: v0.1 requires integer-valued thresholds to
        # canonicalize as floats ("1" -> "1.0"). The vendored copy did not.
        self._parity(self._base(1))

    def test_small_float_threshold_parity(self) -> None:
        self._parity(self._base(1e-6))


class TagScopeTests(unittest.TestCase):
    """MLFLOW_FALSIFY_TAG_SCOPE env var routes descriptive tags off-run."""

    def _with_scope(self, value: str | None):
        # Context manager helper: set env var, restore on exit.
        class _Scope:
            def __init__(self, v: str | None) -> None:
                self.v = v
                self.prev: str | None = None
                self.had: bool = False

            def __enter__(self_inner) -> None:
                self_inner.had = "MLFLOW_FALSIFY_TAG_SCOPE" in os.environ
                self_inner.prev = os.environ.get("MLFLOW_FALSIFY_TAG_SCOPE")
                if self_inner.v is None:
                    os.environ.pop("MLFLOW_FALSIFY_TAG_SCOPE", None)
                else:
                    os.environ["MLFLOW_FALSIFY_TAG_SCOPE"] = self_inner.v

            def __exit__(self_inner, *exc: object) -> None:
                if self_inner.had:
                    os.environ["MLFLOW_FALSIFY_TAG_SCOPE"] = self_inner.prev or ""
                else:
                    os.environ.pop("MLFLOW_FALSIFY_TAG_SCOPE", None)

        return _Scope(value)

    def test_default_scope_emits_all_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(TV_001_YAML, encoding="utf-8")
            with _Chdir(root), self._with_scope(None):
                tags = FalsifyRunContextProvider().tags()
            self.assertEqual(tags.get("prml.manifest_hash"), TV_001_HASH)
            self.assertEqual(tags.get("prml.metric"), "accuracy")
            self.assertEqual(tags.get("prml.threshold"), "0.85")
            self.assertEqual(tags.get("prml.dataset_id"), "imagenet-val-2012")

    def test_experiment_scope_keeps_only_audit_essential(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(TV_001_YAML, encoding="utf-8")
            with _Chdir(root), self._with_scope("experiment"):
                tags = FalsifyRunContextProvider().tags()
            # Audit-essential tags stay
            self.assertEqual(tags.get("prml.manifest_hash"), TV_001_HASH)
            self.assertIn("prml.manifest_path", tags)
            # Descriptive tags lifted off-run
            self.assertNotIn("prml.metric", tags)
            self.assertNotIn("prml.comparator", tags)
            self.assertNotIn("prml.threshold", tags)
            self.assertNotIn("prml.dataset_id", tags)
            self.assertNotIn("prml.version", tags)

    def test_unknown_scope_falls_back_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".prml.yaml").write_text(TV_001_YAML, encoding="utf-8")
            with _Chdir(root), self._with_scope("nonsense"):
                tags = FalsifyRunContextProvider().tags()
            self.assertEqual(tags.get("prml.metric"), "accuracy")

    def test_compute_tags_with_explicit_path(self) -> None:
        from mlflow_falsify.run_context import _compute_tags

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = root / "subdir" / "claim.yaml"
            manifest.parent.mkdir()
            manifest.write_text(TV_001_YAML, encoding="utf-8")
            tags = _compute_tags(manifest)
            self.assertEqual(tags.get("prml.manifest_hash"), TV_001_HASH)


if __name__ == "__main__":
    unittest.main()
