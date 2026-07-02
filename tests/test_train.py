from __future__ import annotations

import pandas as pd

from src.train import build_pipeline, train_model


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "classifier"


def test_train_model_fits_and_predicts():
    df = pd.DataFrame(
        {
            "age": [25, 38, 28, 44, 18],
            "education-num": [10, 13, 10, 14, 8],
            "capital-gain": [0, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0],
            "hours-per-week": [40, 50, 40, 40, 30],
            "workclass": ["Private", "Private", "Local-gov", "Private", "Private"],
            "marital-status": ["Never-married", "Married-civ-spouse"] * 2
            + ["Never-married"],
            "occupation": [
                "Tech-support",
                "Sales",
                "Exec-managerial",
                "Exec-managerial",
                "Sales",
            ],
            "relationship": ["Not-in-family", "Husband"] * 2 + ["Own-child"],
            "race": ["White"] * 5,
            "sex": ["Male", "Male", "Male", "Male", "Female"],
            "native-country": ["United-States"] * 5,
        }
    )
    y = pd.Series(["<=50K", ">50K", ">50K", ">50K", "<=50K"])
    model = train_model(df, y)
    predictions = model.predict(df)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({"<=50K", ">50K"})
