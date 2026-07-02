from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluate import (
    evaluate_baseline,
    evaluate_model,
    save_confusion_matrix,
    save_metrics,
)


def test_evaluate_model_returns_expected_keys():
    df = pd.DataFrame(
        {
            "age": [25, 38, 28, 44],
            "workclass": ["Private", "Private", "Local-gov", "Private"],
        }
    )
    y = pd.Series(["<=50K", ">50K", "<=50K", ">50K"])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(df, y)
    metrics = evaluate_model(model, df, y, name="dummy")
    assert "model" in metrics
    assert "accuracy" in metrics
    assert "weighted_f1" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["model"] == "dummy"


def test_evaluate_baseline_improves_with_constant_class(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate.METRICS_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "age": [25, 38, 28, 44, 35, 50],
            "workclass": ["Private"] * 6,
        }
    )
    y = pd.Series(["<=50K", "<=50K", "<=50K", ">50K", ">50K", ">50K"])
    baseline = evaluate_baseline(df, y, df, y)
    assert baseline["model"] == "baseline"
    assert baseline["accuracy"] == 0.5


def test_save_metrics_creates_json(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate.METRICS_DIR", tmp_path)
    metrics = {"model": "test", "accuracy": 0.9}
    path = save_metrics(metrics, filename="test_metrics.json")
    assert path.exists()


def test_save_confusion_matrix_creates_png(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    metrics = {
        "model": "test",
        "confusion_matrix": [[5, 1], [2, 4]],
    }
    path = save_confusion_matrix(
        metrics, labels=["<=50K", ">50K"], filename="test_cm.png"
    )
    assert path.exists()
    assert path.suffix == ".png"
