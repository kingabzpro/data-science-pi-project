"""Inspect the UCI Adult Census Income dataset before modelling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RANDOM_SEED = 42
TARGET_COLUMN = "class"
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_adult(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Download the Adult dataset from OpenML and cache it locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "adult.csv"
    if path.exists():
        return pd.read_csv(path)

    adult = fetch_openml(
        "adult",
        version=2,
        as_frame=True,
        parser="auto",
    )
    df = adult.frame
    df.to_csv(path, index=False)
    return df


def inspect(df: pd.DataFrame) -> dict:
    """Return a structured inspection summary of the dataset."""
    summary = {
        "shape": df.shape,
        "columns": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "numeric_columns": df.select_dtypes(exclude="object").columns.tolist(),
    }
    return summary


if __name__ == "__main__":
    df = fetch_adult()
    summary = inspect(df)
    for key, value in summary.items():
        print(f"\n{key}:\n{value}")
