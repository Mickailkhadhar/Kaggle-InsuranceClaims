from __future__ import annotations
import pandas as pd
import pytest
from src.transform.feature_engineering import FeatureEngineer
from src.transform.data_validator import DataValidator
from src.utils.config_loader import load_config


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame matching the raw CSV structure before any transformations."""
    return pd.DataFrame({
        "ClaimNumber":               ["C001", "C002", "C003"],
        "DateTimeOfAccident":        ["2020-03-15 14:30:00", "2020-06-01 02:00:00", "2021-01-10 09:00:00"],
        "DateReported":              ["2020-03-18 10:00:00", "2020-06-05 09:00:00", "2021-01-12 11:00:00"],
        "ClaimDescription":          [
            "Patient suffered fracture of lumbar spine",
            "Burn injury due to chemical exposure",
            "No significant injuries noted",
        ],
        "Gender":                    ["M", "F", "M"],
        "MaritalStatus":             ["S", "M", "S"],
        "PartTimeFullTime":          ["F", "P", "F"],
        "WeeklyWages":               [800.0, 400.0, 600.0],
        "InitialIncurredCalimsCost": [5000.0, 1000.0, 3000.0],
        "HoursWorkedPerWeek":        [40.0, 20.0, 35.0],
        "DependentChildren":         ["0", "1", "0"],
        "DependentsOther":           ["0", "0", "1"],
        "DaysWorkedPerWeek":         [5.0, 3.0, 5.0],
        "Age":                       [35.0, 28.0, 45.0],
        "UltimateIncurredClaimCost": [12000.0, 3500.0, 8000.0],
    })


@pytest.fixture
def schema_config() -> dict:
    """Schema config loaded from the actual config file."""
    return load_config("config/schema.yaml")


def _run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in the correct order."""
    df = FeatureEngineer.extract_datetime_features(df)
    df = FeatureEngineer.extract_claim_keywords(df)
    df = FeatureEngineer.drop_text_columns(df)
    df = FeatureEngineer.encode_categoricals(df)
    df = FeatureEngineer.add_wage_to_initial_ratio(df)
    df = FeatureEngineer.add_off_hours_flag(df)
    df = FeatureEngineer.drop_low_signal_features(df)
    return df


def test_pipeline_output_contains_all_schema_features(raw_df: pd.DataFrame, schema_config: dict) -> None:
    """All features declared in schema.yaml must be present in the pipeline output."""
    result = _run_pipeline(raw_df.drop(columns=["UltimateIncurredClaimCost"]))
    expected = schema_config["features"]["numerical"] + schema_config["features"]["categorical"]
    missing = [f for f in expected if f not in result.columns]
    assert missing == [], f"Features missing from pipeline output: {missing}"


def test_pipeline_output_passes_validator(raw_df: pd.DataFrame, schema_config: dict) -> None:
    """Pipeline output must pass DataValidator without raising a SchemaError."""
    result = _run_pipeline(raw_df)
    DataValidator(schema_config).validate(result, is_train=True)


def test_pipeline_output_passes_validator_test_mode(raw_df: pd.DataFrame, schema_config: dict) -> None:
    """Pipeline output on test data (no target) must pass DataValidator in test mode."""
    result = _run_pipeline(raw_df.drop(columns=["UltimateIncurredClaimCost"]))
    DataValidator(schema_config).validate(result, is_train=False)


def test_pipeline_drops_raw_columns(raw_df: pd.DataFrame) -> None:
    """Raw columns that should not reach the model must be absent from the output."""
    result = _run_pipeline(raw_df.drop(columns=["UltimateIncurredClaimCost"]))
    for col in ["ClaimDescription", "Gender", "MaritalStatus", "DateTimeOfAccident",
                "DateReported", "DependentsOther", "DaysWorkedPerWeek",
                "accident_hour", "accident_month", "accident_dayofweek", "report_month"]:
        assert col not in result.columns, f"Column '{col}' should have been dropped"


def test_pipeline_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    """Feature engineering must not modify the original DataFrame."""
    original = raw_df.copy()
    _run_pipeline(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)


def test_pipeline_row_count_preserved(raw_df: pd.DataFrame) -> None:
    """Pipeline must not drop or duplicate rows."""
    result = _run_pipeline(raw_df.drop(columns=["UltimateIncurredClaimCost"]))
    assert len(result) == len(raw_df)
