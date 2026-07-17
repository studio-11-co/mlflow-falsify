"""End-to-end verify-on-run-end tests against a real MLflow file store."""

from __future__ import annotations

import os

import mlflow
import pytest
import yaml

import mlflow_falsify
from mlflow_falsify.verify import evaluate_predicate

MANIFEST = {
    "version": "prml/0.1",
    "claim_id": "01900000-0000-7000-8000-000000000000",
    "created_at": "2026-05-01T12:00:00Z",
    "metric": "accuracy",
    "comparator": ">=",
    "threshold": 0.85,
    "dataset": {"id": "unit-test", "hash": "e" * 64},
    "seed": 42,
    "producer": {"id": "tests"},
}


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    (tmp_path / ".prml.yaml").write_text(yaml.safe_dump(MANIFEST))
    yield tmp_path
    mlflow.set_tracking_uri(None)


def _tags(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id).data.tags


def test_locked_run_pass(tracking):
    with mlflow_falsify.locked_run() as run:
        mlflow.log_metric("accuracy", 0.91)
    tags = _tags(run.info.run_id)
    assert tags["prml.verdict"] == "PASS"
    assert tags["prml.observed"] == "0.91"


def test_locked_run_fail(tracking):
    with mlflow_falsify.locked_run() as run:
        mlflow.log_metric("accuracy", 0.60)
    assert _tags(run.info.run_id)["prml.verdict"] == "FAIL"


def test_metric_never_logged_is_unverified(tracking):
    with mlflow_falsify.locked_run() as run:
        mlflow.log_metric("some_other_metric", 1.0)
    assert _tags(run.info.run_id)["prml.verdict"] == "UNVERIFIED"


def test_manifest_edited_mid_run_is_tampered(tracking):
    with mlflow_falsify.locked_run() as run:
        mlflow.log_metric("accuracy", 0.99)
        # move the goalpost mid-run
        edited = dict(MANIFEST, threshold=0.5)
        (tracking / ".prml.yaml").write_text(yaml.safe_dump(edited))
    assert _tags(run.info.run_id)["prml.verdict"] == "TAMPERED"


def test_verify_run_post_hoc(tracking):
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", 0.86)
    verdict = mlflow_falsify.verify_run(run.info.run_id)
    assert verdict == "PASS"
    assert _tags(run.info.run_id)["prml.verdict"] == "PASS"


def test_strict_mode_raises_on_fail(tracking, monkeypatch):
    monkeypatch.setenv("MLFLOW_FALSIFY_STRICT", "1")
    with pytest.raises(mlflow_falsify.FalsifyVerdictError):
        with mlflow_falsify.locked_run():
            mlflow.log_metric("accuracy", 0.10)


def test_eq_comparator_tolerance():
    # spec 5.1: == is within tolerance (default 1e-9); 0.1+0.2 == 0.3 must pass
    assert evaluate_predicate(0.1 + 0.2, "==", 0.3)
    assert not evaluate_predicate(0.31, "==", 0.3)
    assert evaluate_predicate(0.35, "==", 0.3, tolerance=0.1)


def test_no_manifest_is_unverified(tracking):
    os.remove(tracking / ".prml.yaml")
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", 0.9)
    assert mlflow_falsify.verify_run(run.info.run_id) == "UNVERIFIED"
