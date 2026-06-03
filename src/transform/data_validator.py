from __future__ import annotations
import pandera as pa
import pandas as pd
from loguru import logger


class DataValidator:
    """Validates a DataFrame against expected schema rules defined in schema.yaml.

    Args:
        config: Schema config dict loaded from schema.yaml.
    """
    def __init__(self, config: dict) -> None:
        self._config = config

    def _build_schema(self, has_target: bool = True) -> pa.DataFrameSchema:
        """Build the pandera schema from config.

        Args:
            has_target: Whether to include the target column in the schema.

        Returns:
            DataFrameSchema instance.
        """
        numerical_cols: list[str] = self._config["features"]["numerical"]
        categorical_cols: list[str] = self._config["features"]["categorical"]
        target_col: str = self._config["target_column"]
        id_col: str = self._config["id_column"]
        columns: dict[str, pa.Column] = {}
        columns[id_col] = pa.Column(nullable=False)
        for col in numerical_cols:
            columns[col] = pa.Column(float, coerce=True, nullable=False, checks=[pa.Check.ge(0)])
        for col in categorical_cols:
            columns[col] = pa.Column(nullable=False)
        if has_target:
            columns[target_col] = pa.Column(float, coerce=True, nullable=False, checks=[pa.Check.gt(0)])
        return pa.DataFrameSchema(columns=columns, strict=False)

    def validate(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Validate a DataFrame against the schema.

        Args:
            df: DataFrame to validate.
            is_train: Whether the DataFrame includes the target column.

        Returns:
            Validated DataFrame.

        Raises:
            pa.errors.SchemaError: If the DataFrame does not conform to the schema.
        """
        schema = self._build_schema(has_target=is_train)
        validated = schema.validate(df)
        logger.info(f"Validation passed — {'train' if is_train else 'test'} ({len(df)} rows)")
        return validated
