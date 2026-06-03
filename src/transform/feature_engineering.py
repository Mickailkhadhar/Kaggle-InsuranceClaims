from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Collection of static feature engineering transformations."""

    @staticmethod
    def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataTimeOfAccident and DateReported. 

        Args: 
            df: Input DataFrame with raw datetime columns.

        Returns:
            DataFrame with new datetime features, raw columns dropped
        """
        df = df.copy()
        df["DateTimeOfAccident"] = pd.to_datetime(df["DateTimeOfAccident"])
        df["DateReported"] = pd.to_datetime(df["DateReported"])
        df["days_to_report"] = (df["DateReported"] - df["DateTimeOfAccident"]).dt.days
        df["accident_month"] = df["DateTimeOfAccident"].dt.month.astype(str)
        df["accident_dayofweek"] = df["DateTimeOfAccident"].dt.dayofweek.astype(str)
        df["accident_hour"] = df["DateTimeOfAccident"].dt.hour
        df["report_month"] = df["DateReported"].dt.month.astype(str)
        df = df.drop(columns=["DateTimeOfAccident", "DateReported"])
        logger.info("Datetime features extracted")
        return df
    
    @staticmethod
    def drop_text_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop text columns that are not useful yet for modeling.

        Args:
            df: Input DataFrame with raw text columns.
        Returns:
            DataFrame with text columns dropped.
        """
        df = df.copy()
        text_cols = ["ClaimDescription"]
        df = df.drop(columns=text_cols)
        logger.info(f"Dropped text columns: {text_cols}")
        return df
    
    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Encode low-cardinality categorical columns to integers.

        Mappings:
            Gender: M=0, F=1, U=2
            MaritalStatus: S=0, M=1, U=2
            PartTimeFullTime: P=0, F=1

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with encoded columns.
        """
        df = df.copy()
        df["Gender"] = df["Gender"].map({"M": "0", "F": "1", "U": "2"})
        df["MaritalStatus"] = df["MaritalStatus"].map({"S": "0", "M": "1", "U": "2"})
        df["PartTimeFullTime"] = df["PartTimeFullTime"].map({"P": "0", "F": "1"})
        logger.info("Categorical columns encoded")
        return df

    @staticmethod
    def add_wage_to_initial_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """Add wage-to-initial-reserve ratio to deconfound WeeklyWages from gender.

        Divides WeeklyWages by InitialIncurredCalimsCost to produce a relative
        severity signal that is not distorted by the gender pay gap. A low-wage
        claimant with a high initial reserve will have a low ratio, correctly
        signalling elevated severity independent of absolute wage level.

        Args:
            df: Input DataFrame containing WeeklyWages and InitialIncurredCalimsCost.

        Returns:
            DataFrame with new column wage_to_initial_ratio added.
        """
        df = df.copy()
        df["wage_to_initial_ratio"] = df["WeeklyWages"] / np.maximum(df["InitialIncurredCalimsCost"], 1)
        logger.info("Feature wage_to_initial_ratio added")
        return df

    @staticmethod
    def add_off_hours_flag(df: pd.DataFrame) -> pd.DataFrame:
        """Add binary flag for accidents occurring outside standard working hours.

        .. deprecated::
            is_off_hours has been removed from the modelling feature set (schema.yaml)
            after SHAP analysis showed insufficient signal. This method is kept for
            reference and reproducibility but its output column is no longer used.

        Off-hours is defined as outside 08:00–17:59, covering overnight shifts,
        early morning, and weekend patterns where injury severity tends to be higher
        due to reduced supervision and fatigue.

        Args:
            df: Input DataFrame containing accident_hour column.

        Returns:
            DataFrame with new binary column is_off_hours (1 = off-hours, 0 = standard).
        """
        df = df.copy()
        df["is_off_hours"] = (~df["accident_hour"].between(8, 17)).astype(int)
        logger.info("Feature is_off_hours added")
        return df

    @staticmethod
    def extract_claim_keywords(df: pd.DataFrame) -> pd.DataFrame:
        """Extract binary keyword flags from ClaimDescription text.

        Each flag captures an injury pattern with actuarially distinct severity profiles.
        Must be called before drop_text_columns() since it consumes ClaimDescription.

        Keyword groups:
            is_back:       spine/lumbar/cervical injuries — high chronic development.
            is_head:       head/brain trauma — potential long-tail severity.
            is_surgery:    surgical intervention — strong predictor of large losses.
            is_fracture:   broken bones — moderate-to-high severity.
            is_permanent:  permanent disability / chronic — highest development factor.
            is_burn:       burn / chemical injuries — high medical cost and treatment.
            is_contusion:  soft tissue injuries — moderate, high frequency.

        Args:
            df: Input DataFrame containing ClaimDescription column.

        Returns:
            DataFrame with 7 new binary integer columns added.
        """
        _KEYWORDS: dict[str, list[str]] = {
            "is_back": ["back", "spine", "lumbar", "cervical"],
            "is_head": ["head", "skull", "brain", "concussion"],
            "is_surgery": ["surgery", "surgical", "operation", "amputation"],
            "is_fracture": ["fracture", "broken", "break"],
            "is_permanent": ["permanent", "disability", "chronic"],
            "is_burn": ["burn", "fire", "chemical"],
            "is_contusion": ["contusion", "bruise", "strain", "sprain"],
        }
        df = df.copy()
        desc = df["ClaimDescription"].str.lower().fillna("")
        for col, terms in _KEYWORDS.items():
            df[col] = desc.str.contains("|".join(terms), regex=True).astype(int)
        logger.info(f"Claim keyword features extracted: {list(_KEYWORDS.keys())}")
        return df

    @staticmethod
    def drop_low_signal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Drop features identified as near-zero SHAP importance with no actuarial justification.

        Dropped features:
            - report_month, accident_month, accident_dayofweek, accident_hour: no causal
              link between calendar/time-of-day and ultimate claim severity.
              accident_hour is consumed by add_off_hours_flag before being dropped.
            - Gender, MaritalStatus, PartTimeFullTime: confounding absorbed by
              wage_to_initial_ratio; near-zero SHAP contribution.
            - DependentChildren, DependentsOther: no signal observed.
            - DaysWorkedPerWeek: signal absorbed by HoursWorkedPerWeek.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with low-signal columns dropped.
        """
        df = df.copy()
        cols_to_drop = [
            "report_month",
            "accident_month",
            "accident_dayofweek",
            "accident_hour",
            "Gender",
            "MaritalStatus",
            "DependentsOther",
            "DaysWorkedPerWeek",
        ]
        existing = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing)
        logger.info(f"Dropped low-signal features: {existing}")
        return df
        