# Project Rules

## Research
- Use Olostep for current facts, benchmarks, papers, package comparisons, and external research.
- Use Context7 before writing code that depends on an external library or SDK.
- Prefer official documentation, source repositories, and primary papers.

## Data Science
- Inspect schema, data types, missing values, duplicates, leakage risks, and label balance before modelling.
- Define the evaluation metric and baseline before training.
- Use a reproducible train-validation-test split or cross-validation strategy.
- Record the random seed, features, preprocessing steps, model configuration, and metrics.
- Never claim a model improved without comparing it to a baseline.

## Engineering
- Keep reusable logic in src/ rather than notebooks.
- Add tests when changing preprocessing or evaluation behaviour.
- Do not silently modify labels, splits, or preprocessing rules.

## Completion
- Run formatting, linting, and tests before declaring work complete.
- Summarize changed files, metrics, tests run, and known limitations.

Run the basic project checks directly with uv:
uv run ruff format .
uv run ruff check .
uv run pytest -q

For each new experiment, Pi should follow this order:
1. Inspect the dataset.
2. Check missing values, duplicates, schema, and leakage risks.
3. Define the target, metric, baseline, and split strategy.
4. Build preprocessing in src/data.py.
5. Train a baseline in src/train.py.
6. Evaluate it in src/evaluate.py.
7. Save figures and metrics in reports/.
8. Add or update tests.
9. Run Ruff and pytest.
10. Record the decision and result in docs/decisions/.

This keeps notebooks useful for exploration while ensuring the actual pipeline is reproducible, testable, and easy for Pi to maintain.
