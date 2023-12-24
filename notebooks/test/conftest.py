import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def notebooks_path():
    notebooks_dir = Path(__file__).parent.parent
    with patch.dict(os.environ, {"PYXSURF_NOTEBOOKS_PATH": str(notebooks_dir)}):
        yield notebooks_dir
