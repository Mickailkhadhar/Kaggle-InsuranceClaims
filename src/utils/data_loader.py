from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


class DataLoader:
    """Loads raw CSV datasets from paths defined in the paths config.

    Args:
        config: Parsed content of config/paths.yaml.
    """

    def __init__(self, config: dict) -> None:
        self._config = config["data"]
        self._root = Path(".")

    def _load(self, key: str) -> pd.DataFrame:
        """Load a CSV file by its config key.

        Args:
            key: Key in config["data"], e.g. "train", "test", "sample_submission".

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        path = self._root / self._config[key]
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        logger.info(f"Loading '{key}' from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded '{key}': shape={df.shape}")
        return df

    def load_train(self) -> pd.DataFrame:
        """Load the training dataset.

        Returns:
            Training DataFrame.
        """
        return self._load("train")

    def load_test(self) -> pd.DataFrame:
        """Load the test dataset.

        Returns:
            Test DataFrame.
        """
        return self._load("test")

    def load_sample_submission(self) -> pd.DataFrame:
        """Load the sample submission file.

        Returns:
            Sample submission DataFrame.
        """
        return self._load("sample_submission")