from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    preprocess,
    split_features_target,
    train_test_split_data,
)


def test_preprocess_drops_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 5], TARGET_COLUMN: [0, 0, 1]})
    cleaned = preprocess(df)
    assert len(cleaned) == 2


def test_preprocess_drops_all_empty_rows():
    df = pd.DataFrame({"a": [1, None], "b": [2, None], TARGET_COLUMN: [0, None]})
    cleaned = preprocess(df)
    assert len(cleaned) == 1


def test_preprocess_preserves_order_and_resets_index():
    df = pd.DataFrame({"age": [1, 2, 1], TARGET_COLUMN: ["<=50K", ">50K", "<=50K"]})
    cleaned = preprocess(df)
    assert cleaned.index.tolist() == [0, 1]
    assert cleaned["age"].tolist() == [1, 2]


def test_preprocess_drops_leakage_columns():
    df = pd.DataFrame(
        {
            "age": [25, 38],
            "education-num": [10, 13],
            "education": ["HS-grad", "Bachelors"],
            "fnlwgt": [1000, 2000],
            TARGET_COLUMN: ["<=50K", ">50K"],
        }
    )
    cleaned = preprocess(df)
    assert "education" not in cleaned.columns
    assert "fnlwgt" not in cleaned.columns
    assert "age" in cleaned.columns


def test_preprocess_imputes_missing_categorical():
    df = pd.DataFrame(
        {
            "workclass": ["Private", None],
            "occupation": ["Tech-support", None],
            "age": [25, 38],
            "education-num": [10, 13],
            TARGET_COLUMN: ["<=50K", ">50K"],
        }
    )
    cleaned = preprocess(df)
    assert cleaned["workclass"].isna().sum() == 0
    assert cleaned["occupation"].isna().sum() == 0
    assert "Missing" in cleaned["workclass"].values


def test_split_features_target_separates_target():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], TARGET_COLUMN: [0, 1]})
    X, y = split_features_target(df, target=TARGET_COLUMN)
    assert list(X.columns) == ["a", "b"]
    assert list(y) == [0, 1]


def test_split_features_target_missing_target_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        split_features_target(df, target=TARGET_COLUMN)


def test_train_test_split_data_stratifies():
    df = pd.DataFrame(
        {
            "age": list(range(1000)),
            "workclass": ["Private"] * 1000,
            TARGET_COLUMN: ["<=50K"] * 800 + [">50K"] * 200,
        }
    )
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert set(y_train.unique()) == {"<=50K", ">50K"}
    assert set(y_test.unique()) == {"<=50K", ">50K"}


def test_feature_lists_are_disjoint_and_include_target():
    assert not set(NUMERIC_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert TARGET_COLUMN not in NUMERIC_COLUMNS
    assert TARGET_COLUMN not in CATEGORICAL_COLUMNS
