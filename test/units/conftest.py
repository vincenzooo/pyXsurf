from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def test_units_path(test_path: Path):
    return test_path / "units"
