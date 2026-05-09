"""
Pytest fixtures for extract module integration tests.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.fixture
def bai_path() -> Path:
    """
    Return the integration-test BAI path from the environment.

    Returns:
        Path:
            The path referenced by the `TEST_BAI_PATH`
            environment variable.
    """

    value = os.environ.get("TEST_BAI_PATH")
    if not value:
        pytest.fail("TEST_BAI_PATH must be set for extract tests")

    path = Path(value)
    if not path.exists():
        pytest.fail(f"TEST_BAI_PATH does not exist: {path}")

    return path
