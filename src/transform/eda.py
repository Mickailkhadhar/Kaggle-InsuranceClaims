from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


class EDA:
    """Performs exploratory data analysis on a DataFrame.

    Args:
        df: Input DataFrame to analyze.
        target: Name of the target column.
    """

    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self._df = df
        self._target = target

    def summary(self) -> pd.DataFrame:
        """Return a summary of dtypes, null counts and unique counts per column.

        Returns:
            DataFrame with one row per column.
        """
        summary = pd.DataFrame({
            "dtype": self._df.dtypes,
            "null_count": self._df.isnull().sum(),
            "null_pct": (self._df.isnull().mean() * 100).round(2),
            "n_unique": self._df.nunique(),
        })
        logger.info(f"Summary computed for {len(summary)} columns")
        return summary

    def target_distribution(self) -> None:
        """Plot the target distribution (raw and log-transformed).

        Returns:
            None
        """
        _, axes = plt.subplots(1, 2, figsize=(12, 10))
        sns.histplot(self._df[self._target], ax=axes[0], kde=True)
        axes[0].set_title(f"{self._target} — raw")
        sns.histplot(np.log1p(self._df[self._target]), ax=axes[1], kde=True, color="coral")
        axes[1].set_title(f"{self._target} — log1p")
        plt.tight_layout()
        plt.show()

    def missing_report(self) -> pd.DataFrame:
        """Return columns with missing values sorted by null percentage.

        Returns:
            DataFrame of columns with at least one missing value.
        """
        report = (
            self._df.isnull()
            .mean()
            .mul(100)
            .round(2)
            .rename("null_pct")
            .to_frame()
            .query("null_pct > 0")
            .sort_values("null_pct", ascending=False)
        )
        logger.info(f"{len(report)} columns with missing values")
        return report

    def cardinality_report(self) -> pd.DataFrame:
        """Return cardinality of categorical columns sorted descending.

        Returns:
            DataFrame with unique value counts per categorical column.
        """
        cat_cols = self._df.select_dtypes(include="object").columns
        return (
            self._df[cat_cols]
            .nunique()
            .rename("n_unique")
            .to_frame()
            .sort_values("n_unique", ascending=False)
        )

    def correlation_matrix(self) -> None:
        """Plot a heatmap of correlations between numerical features.

        Returns:
            None
        """
        num_cols = self._df.select_dtypes(include="number").columns
        corr = self._df[num_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.show()

    def categorical_vs_target(self, col: str) -> None:
        """Plot boxplot of target distribution per category.

        Args:
            col: Categorical column name.

        Returns:
            None
        """
        order = sorted(self._df[col].dropna().unique().tolist())
        plt.figure(figsize=(12, 10))
        sns.boxplot(data=self._df, x=col, y=self._target, order=order)
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.title(f"{self._target} by {col}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def numerical_vs_target(df: pd.DataFrame, col: str, target: str, clip_percentile: float = 99.0) -> None:
        """Plot scatter of a numerical feature vs target, clipped for display only.

        Args:
            df: Input DataFrame.
            col: Numerical column name.
            target: Target column name.
            clip_percentile: Upper percentile to clip values for visualization.

        Returns:
            None
        """
        clip_val = np.percentile(df[target].dropna(), clip_percentile)
        plot_df = df[df[target] <= clip_val]
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=plot_df, x=col, y=target, alpha=0.3)
        plt.title(f"{col} vs {target} (clipped at p{int(clip_percentile)})")
        plt.tight_layout()
        plt.show()
