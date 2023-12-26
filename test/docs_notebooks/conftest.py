import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def notebooks_path():
    project_root = Path(__file__).parent.parent.parent
    notebooks_dir = project_root / "docs" / "source" / "notebooks"
    with patch.dict(os.environ, {"PYXSURF_NOTEBOOKS_PATH": str(notebooks_dir)}):
        yield notebooks_dir


@pytest.fixture(scope="module")
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp
