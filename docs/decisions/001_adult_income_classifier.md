# Decision: UCI Adult Income classifier

## Context

Build a reproducible binary classification pipeline to predict whether an individual's income exceeds $50K/year using the UCI Adult Census Income dataset.

## Dataset

- Source: UCI Machine Learning Repository / OpenML (`adult`, version 2)
- Rows: 48,842
- Target: `class` (`<=50K`, `>50K`)
- Class balance: ~76% `<=50K`, ~24% `>50K`
- Missing values: `workclass` (2,799), `occupation` (2,809), `native-country` (857)
- Duplicate rows: 52

## Leakage handling

- `fnlwgt` is a survey weight, not a predictor — dropped.
- `education` is redundant with the ordinal `education-num` — dropped to avoid multicollinearity.

## Preprocessing

- Drop exact duplicates and fully-empty rows.
- Impute missing categoricals with a constant `"Missing"` category.
- One-hot encode categorical features.
- Standard-scale numeric features.

## Model

- Random Forest classifier (`n_estimators=200`, `max_depth=16`, `min_samples_leaf=2`).
- All preprocessing and modelling inside a single scikit-learn `Pipeline`.

## Evaluation

- Stratified train/test split (80/20) with `random_state=42`.
- Metrics: accuracy, weighted F1, macro F1, confusion matrix.
- Baseline: most-frequent class classifier.

## Results on test set

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (most frequent) | 0.7606 | 0.6572 | 0.4320 |
| Random Forest | 0.8607 | 0.8523 | 0.7870 |

The Random Forest improves over the baseline on all reported metrics.

## Files changed

- `src/data.py`: loading, cleaning, imputation, stratified split.
- `src/train.py`: pipeline construction and model training.
- `src/evaluate.py`: metric computation, baseline comparison, figure generation.
- `src/main.py`: end-to-end orchestration.
- `notebooks/01_inspect_dataset.py`: initial dataset inspection.
- `tests/test_data.py`, `tests/test_train.py`, `tests/test_evaluate.py`: unit tests.
- `reports/metrics/metrics.json`, `reports/figures/confusion_matrix.png`: outputs.
