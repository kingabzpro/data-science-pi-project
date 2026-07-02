"""End-to-end reproducible training and evaluation pipeline for Adult income classification."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import (
    fetch_and_cache_raw_data,
    preprocess,
    save_processed,
    train_test_split_data,
)
from src.evaluate import (
    evaluate_baseline,
    evaluate_model,
    save_confusion_matrix,
    save_metrics,
)
from src.train import save_model, train_model

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def run() -> dict:
    """Run the full ML workflow and return test metrics."""
    # 1. Load / fetch data
    raw_df = fetch_and_cache_raw_data()

    # 2. Preprocess
    processed_df = preprocess(raw_df)
    save_processed(processed_df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split_data(processed_df)

    # 4. Train
    model = train_model(X_train, y_train)
    save_model(model)

    # 5. Evaluate
    test_metrics = evaluate_model(model, X_test, y_test, name="random_forest")
    baseline_metrics = evaluate_baseline(X_train, y_train, X_test, y_test)

    combined = {
        "random_state": 42,
        "test_size": 0.2,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
    }

    save_metrics(combined)
    save_confusion_matrix(
        test_metrics, labels=sorted(y_test.unique()), filename="confusion_matrix.png"
    )

    # Print a concise summary.
    print(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":
    run()
