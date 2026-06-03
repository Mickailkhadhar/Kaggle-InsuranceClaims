from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.data_loader import DataLoader


@pytest.fixture
def config(tmp_path: Path) -> dict:
    """Build a config pointing to temporary CSV files."""
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    submission = tmp_path / "sample_submission.csv"

    pd.DataFrame({"id": [1, 2], "loss": [100.0, 200.0]}).to_csv(train, index=False)
    pd.DataFrame({"id": [3, 4]}).to_csv(test, index=False)
    pd.DataFrame({"id": [3, 4], "loss": [0.0, 0.0]}).to_csv(submission, index=False)

    return {
        "data": {
            "train": str(train),
            "test": str(test),
            "sample_submission": str(submission),
        }
    }


def test_load_train(config: dict) -> None:
    df = DataLoader(config).load_train()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_test(config: dict) -> None:
    df = DataLoader(config).load_test()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_sample_submission(config: dict) -> None:
    df = DataLoader(config).load_sample_submission()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_raises_if_file_missing() -> None:
    config = {"data": {"train": "nonexistent.csv", "test": "", "sample_submission": ""}}
    with pytest.raises(FileNotFoundError):
        DataLoader(config).load_train()
