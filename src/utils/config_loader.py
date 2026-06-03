from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a dictionary.
    """
    with open(Path(path)) as f:
        return yaml.safe_load(f)
