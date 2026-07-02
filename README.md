# Adult Income Classification

A reproducible Python machine-learning project for binary classification: predict whether an individual's annual income exceeds $50K using the [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) dataset.

## Quick start

```bash
# Fetch data, train model, and generate reports
uv run python -m src.main

# Run checks
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## Project structure

```text
src/
  data.py       — loading, cleaning, imputation, stratified train/test split
  train.py      — scikit-learn Pipeline with preprocessing + Random Forest
  evaluate.py   — accuracy, weighted F1, macro F1, confusion matrix, baseline
  main.py       — end-to-end orchestration

tests/          — unit tests for data, training, and evaluation
notebooks/      — exploratory inspection scripts
reports/        — metrics JSON and confusion matrix figures
models/         — persisted trained model
data/           — raw and processed datasets
```

## Reproducibility

- Fixed random seed: `42`
- Stratified train/test split: 80/20
- All randomised estimators accept `random_state=42`
- Dependencies pinned in `uv.lock`

## Latest results

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (most frequent) | 0.7606 | 0.6572 | 0.4320 |
| Random Forest | 0.8607 | 0.8523 | 0.7870 |

Confusion matrix on the test set:

```text
[[7070  352]
 [1007 1329]]
```

## Web interface

Launch the Streamlit app:

```bash
uv run streamlit run app.py
```

## Dataset reference

Kohavi, R., & Becker, B. (1996). UCI Machine Learning Repository: Adult Data Set. https://archive.ics.uci.edu/ml/datasets/adult
