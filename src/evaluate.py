from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.data import RANDOM_SEED

matplotlib.use("Agg")

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


def evaluate_model(model, X, y, name: str = "model") -> dict:
    """Compute classification metrics for a fitted model."""
    predictions = model.predict(X)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, predictions)),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def evaluate_baseline(X_train, y_train, X, y) -> dict:
    """Train and evaluate a most-frequent class baseline."""
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    baseline.fit(X_train, y_train)
    return evaluate_model(baseline, X, y, name="baseline")


def save_metrics(metrics: dict, filename: str = "metrics.json") -> Path:
    """Persist metrics as JSON under reports/metrics/."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_confusion_matrix(
    metrics: dict,
    labels: list[str] | None = None,
    filename: str = "confusion_matrix.png",
) -> Path:
    """Render and save a confusion matrix heatmap under reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename

    cm = np.array(metrics["confusion_matrix"])
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {metrics['model']}")

    # Annotate cells.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
