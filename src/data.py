from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TARGET_COLUMN = "class"

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Columns with a derivation or survey-weight relationship to the target that
# should not be used as predictors.
DROPPED_COLUMNS = ["education", "fnlwgt"]

CATEGORICAL_COLUMNS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def load_raw_data(filename: str = "adult.csv") -> pd.DataFrame:
    """Load the original dataset from data/raw/ without modifying it."""
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run notebooks/01_inspect_dataset.py to fetch it."
        )
    return pd.read_csv(path)


def fetch_and_cache_raw_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Download the UCI Adult dataset from OpenML and cache it locally."""
    from sklearn.datasets import fetch_openml

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "adult.csv"
    if path.exists():
        return pd.read_csv(path)

    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame
    df.to_csv(path, index=False)
    return df


def _impute_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing categorical values with a constant 'Missing' category."""
    df = df.copy()
    present = [c for c in columns if c in df.columns]
    if present:
        imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        df[present] = imputer.fit_transform(df[present])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Adult dataset: drop duplicates, leakage columns, and impute."""
    cleaned = df.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)

    # Remove leakage / derived columns.
    cleaned = cleaned.drop(columns=[c for c in DROPPED_COLUMNS if c in cleaned.columns])

    # Keep only predictors and target.
    keep = [
        c
        for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
        if c in cleaned.columns
    ]
    cleaned = cleaned[keep]

    # Impute missing categoricals so downstream encoders see no NaNs.
    cleaned = _impute_categorical(cleaned, CATEGORICAL_COLUMNS)

    return cleaned


def split_features_target(
    df: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {list(df.columns)}")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def _safe_stratify(y: pd.Series, stratify: bool) -> pd.Series | None:
    """Stratify only when every class has at least two samples."""
    if not stratify:
        return None
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y


def train_test_split_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproducibly split data into stratified train and test sets."""
    X, y = split_features_target(df, target=target)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=_safe_stratify(y, stratify),
    )


def save_processed(df: pd.DataFrame, filename: str = "processed.csv") -> Path:
    """Persist a cleaned dataset to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path
