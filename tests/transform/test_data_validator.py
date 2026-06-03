from __future__ import annotations
import pandera as pa
import pandas as pd
import pytest
from src.transform.data_validator import DataValidator


@pytest.fixture
def schema_config() -> dict:
    """Minimal schema config mirroring schema.yaml structure (post-feature-engineering)."""
    return {
        "target_column": "UltimateIncurredClaimCost",
        "id_column": "ClaimNumber",
        "features": {
            "numerical": [
                "Age",
                "WeeklyWages",
                "HoursWorkedPerWeek",
                "InitialIncurredCalimsCost",
                "days_to_report",
                "wage_to_initial_ratio",
                "is_back",
                "is_head",
                "is_surgery",
                "is_fracture",
                "is_burn",
                "is_contusion",
            ],
            "categorical": [],
        },
    }


@pytest.fixture
def valid_train_df() -> pd.DataFrame:
    """One valid train row with all expected post-FE columns."""
    return pd.DataFrame({
        "ClaimNumber": ["C001"],
        "Age": [35.0],
        "WeeklyWages": [800.0],
        "HoursWorkedPerWeek": [40.0],
        "InitialIncurredCalimsCost": [5000.0],
        "days_to_report": [3.0],
        "wage_to_initial_ratio": [0.16],
        "is_back": [0.0],
        "is_head": [0.0],
        "is_surgery": [0.0],
        "is_fracture": [0.0],
        "is_burn": [0.0],
        "is_contusion": [0.0],
        "UltimateIncurredClaimCost": [12000.0],
    })


def test_validate_train_passes(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    result = DataValidator(schema_config).validate(valid_train_df, is_train=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_validate_test_passes(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    test_df = valid_train_df.drop(columns=["UltimateIncurredClaimCost"])
    result = DataValidator(schema_config).validate(test_df, is_train=False)
    assert isinstance(result, pd.DataFrame)


def test_validate_fails_on_negative_numerical(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    invalid_df = valid_train_df.copy()
    invalid_df["Age"] = [-1.0]
    with pytest.raises(pa.errors.SchemaError):
        DataValidator(schema_config).validate(invalid_df, is_train=True)


def test_validate_fails_on_null_numerical(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    invalid_df = valid_train_df.copy()
    invalid_df["WeeklyWages"] = [None]
    with pytest.raises(pa.errors.SchemaError):
        DataValidator(schema_config).validate(invalid_df, is_train=True)


def test_validate_fails_on_zero_target(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    invalid_df = valid_train_df.copy()
    invalid_df["UltimateIncurredClaimCost"] = [0.0]
    with pytest.raises(pa.errors.SchemaError):
        DataValidator(schema_config).validate(invalid_df, is_train=True)


def test_validate_fails_on_null_keyword_feature(schema_config: dict, valid_train_df: pd.DataFrame) -> None:
    invalid_df = valid_train_df.copy()
    invalid_df["is_back"] = [None]
    with pytest.raises(pa.errors.SchemaError):
        DataValidator(schema_config).validate(invalid_df, is_train=True)