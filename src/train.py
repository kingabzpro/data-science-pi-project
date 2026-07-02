from __future__ import annotations

import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RANDOM_SEED

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def build_pipeline(random_state: int = RANDOM_SEED) -> Pipeline:
    """Build a reproducible preprocessing + model pipeline for the Adult dataset."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(X_train, y_train, random_state: int = RANDOM_SEED) -> Pipeline:
    """Train the preprocessing + Random Forest pipeline."""
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, filename: str = "model.joblib") -> Path:
    """Persist a fitted model to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str = "model.joblib"):
    """Load a persisted model from models/."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
