from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def test_path():
    return Path(__file__).parent.parent
