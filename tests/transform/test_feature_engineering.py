from __future__ import annotations
import pandas as pd
import pytest
from src.transform.feature_engineering import FeatureEngineer


KEYWORD_COLUMNS = ["is_back", "is_head", "is_surgery", "is_fracture", "is_permanent", "is_burn", "is_contusion"]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame mirroring the raw CSV structure."""
    return pd.DataFrame({
        "DateTimeOfAccident": ["2020-03-15 00:00:00", "2020-06-01 02:00:00"],
        "DateReported":       ["2020-03-18 00:00:00", "2020-06-05 09:00:00"],
        "ClaimDescription":   ["Patient suffered fracture of lumbar spine", "No injuries"],
        "Gender":             ["M", "F"],
        "MaritalStatus":      ["S", "M"],
        "PartTimeFullTime":   ["F", "P"],
        "WeeklyWages":        [800.0, 400.0],
        "InitialIncurredCalimsCost": [5000.0, 0.0],
        "HoursWorkedPerWeek": [40.0, 20.0],
        "DependentChildren":  ["0", "1"],
        "DependentsOther":    ["0", "0"],
        "DaysWorkedPerWeek":  [5.0, 3.0],
        "Age":                [35.0, 28.0],
    })


# ── extract_datetime_features ─────────────────────────────────────────────────

def test_extract_datetime_features_creates_days_to_report(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_datetime_features(raw_df)
    assert "days_to_report" in result.columns
    assert result["days_to_report"].iloc[0] == 3


def test_extract_datetime_features_creates_accident_hour(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_datetime_features(raw_df)
    assert "accident_hour" in result.columns
    assert result["accident_hour"].iloc[0] == 0


def test_extract_datetime_features_drops_raw_columns(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_datetime_features(raw_df)
    assert "DateTimeOfAccident" not in result.columns
    assert "DateReported" not in result.columns


def test_extract_datetime_features_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    original_cols = set(raw_df.columns)
    FeatureEngineer.extract_datetime_features(raw_df)
    assert set(raw_df.columns) == original_cols


# ── extract_claim_keywords ────────────────────────────────────────────────────

def test_extract_claim_keywords_creates_all_columns(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_claim_keywords(raw_df)
    for col in KEYWORD_COLUMNS:
        assert col in result.columns


def test_extract_claim_keywords_detects_fracture(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_claim_keywords(raw_df)
    assert result["is_fracture"].iloc[0] == 1


def test_extract_claim_keywords_detects_back(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_claim_keywords(raw_df)
    assert result["is_back"].iloc[0] == 1


def test_extract_claim_keywords_no_false_positives(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_claim_keywords(raw_df)
    assert result["is_fracture"].iloc[1] == 0
    assert result["is_back"].iloc[1] == 0


def test_extract_claim_keywords_case_insensitive(raw_df: pd.DataFrame) -> None:
    df = raw_df.copy()
    df["ClaimDescription"] = ["FRACTURE of SPINE", "nothing"]
    result = FeatureEngineer.extract_claim_keywords(df)
    assert result["is_fracture"].iloc[0] == 1
    assert result["is_back"].iloc[0] == 1


def test_extract_claim_keywords_handles_null_description(raw_df: pd.DataFrame) -> None:
    df = raw_df.copy()
    df["ClaimDescription"] = [None, None]
    result = FeatureEngineer.extract_claim_keywords(df)
    for col in KEYWORD_COLUMNS:
        assert result[col].sum() == 0


def test_extract_claim_keywords_preserves_claim_description(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.extract_claim_keywords(raw_df)
    assert "ClaimDescription" in result.columns


def test_extract_claim_keywords_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    original_cols = set(raw_df.columns)
    FeatureEngineer.extract_claim_keywords(raw_df)
    assert set(raw_df.columns) == original_cols


# ── add_wage_to_initial_ratio ─────────────────────────────────────────────────

def test_add_wage_to_initial_ratio_creates_column(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.add_wage_to_initial_ratio(raw_df)
    assert "wage_to_initial_ratio" in result.columns


def test_add_wage_to_initial_ratio_correct_value(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.add_wage_to_initial_ratio(raw_df)
    assert abs(result["wage_to_initial_ratio"].iloc[0] - 800.0 / 5000.0) < 1e-9


def test_add_wage_to_initial_ratio_avoids_zero_division(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.add_wage_to_initial_ratio(raw_df)
    assert result["wage_to_initial_ratio"].iloc[1] == 400.0


# ── drop_low_signal_features ──────────────────────────────────────────────────

def test_drop_low_signal_features_removes_expected_columns(raw_df: pd.DataFrame) -> None:
    df = FeatureEngineer.extract_datetime_features(raw_df)
    result = FeatureEngineer.drop_low_signal_features(df)
    for col in ["Gender", "MaritalStatus", "accident_hour", "report_month",
                "accident_month", "accident_dayofweek", "DependentsOther", "DaysWorkedPerWeek"]:
        assert col not in result.columns


def test_drop_low_signal_features_tolerates_missing_columns(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer.drop_low_signal_features(raw_df)
    assert isinstance(result, pd.DataFrame)
