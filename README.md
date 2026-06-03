# Actuarial Loss Prediction

Kaggle competition — predict `UltimateIncurredClaimCost` for insurance claims.
Technical interview project for an insurance company.

---

## Stack

- Python 3.12
- uv (package manager + venv)
- pandas, scikit-learn, xgboost
- pandera (data validation)
- optuna (hyperparameter tuning)
- loguru (logging)
- seaborn, matplotlib (EDA plots)
- pytest (unit + integration tests)
- flake8 (linting)

---

## Project Structure

```
config/
  paths.yaml            # data and output file paths
  schema.yaml           # feature lists, target column, id column
  model.yaml            # xgboost params, optuna search space, CV config
src/
  utils/
    config_loader.py    # load_config(path) -> dict
    data_loader.py      # DataLoader class: load_train, load_test, load_sample_submission
  transform/
    eda.py              # EDA class: summary, target_distribution, missing_report,
                        #   cardinality_report, correlation_matrix,
                        #   categorical_vs_target, numerical_vs_target
    feature_engineering.py  # FeatureEngineer class (static methods):
                             #   extract_datetime_features, drop_text_columns, encode_categoricals
    data_validator.py   # DataValidator class: validate(df, is_train) using pandera + schema.yaml
  training/
    trainer.py          # Trainer class: fit(X, y) -> XGBRegressor (params from model.yaml)
    tuner.py            # Tuner class: tune(X, y) -> best_params, trials_report() -> DataFrame
  evaluate/
    metrics.py          # rmse(y_true, y_pred) -> float
tests/
  utils/
    test_data_loader.py
  transform/
    test_data_validator.py
    test_feature_engineering.py
  train/
    test_trainer.py
    test_tuner.py
  evaluate/
    test_metrics.py
  integration/
    test_pipeline.py
explo.ipynb             # EDA notebook
model.ipynb             # modelling notebook (tuning + training + evaluation)
main.py                 # entrypoint (to be built)
Taskfile.yml            # task runner
pyproject.toml          # uv project config
```

---

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
task create-venv
task install        # uv pip install -e .
task update-venv    # uv sync

# Run tests
task unit-tests
task integration-tests
task test              # unit + integration
```

---

## Taskfile Commands

| Command | Description |
|---|---|
| `task create-venv` | Create virtualenv with uv |
| `task update-venv` | Sync deps from lockfile |
| `task install` | Install project as editable package |
| `task unit-tests` | Run unit tests only |
| `task integration-tests` | Run integration tests only |
| `task test` | Run all tests (unit + integration) |

---

## Config: schema.yaml

```yaml
target_column: UltimateIncurredClaimCost
id_column: ClaimNumber
features:
  categorical: [Gender, MaritalStatus, DependentChildren, DependentsOther,
                PartTimeFullTime, accident_month, accident_dayofweek, report_month]
  numerical: [Age, WeeklyWages, HoursWorkedPerWeek, DaysWorkedPerWeek,
              InitialIncurredCalimsCost, days_to_report, accident_hour]
```

---

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs on every push and pull request:

1. **Lint** — flake8 on `src/` and `tests/` (max line length 120)
2. **Unit tests** — all tests except `tests/integration/`
3. **Integration tests** — `tests/integration/` only (runs the full feature engineering pipeline against the live `config/schema.yaml`)

Unit and integration jobs both depend on lint passing first.

---

## Data Submission: 

I only let : 
1- the first submission with a dumb model. 
2- the best performer model that I have, ranking 1st tier in the leaderboard.

