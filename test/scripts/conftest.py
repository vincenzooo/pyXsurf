from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def test_scripts_path(test_path: Path):
    return test_path / "scripts"
