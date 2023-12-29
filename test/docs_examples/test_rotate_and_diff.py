import subprocess
import sys
from pathlib import Path


def test_main(examples_path: Path):
    """Run system Python interpreter on the test script corresponding to this test."""
    subprocess.run(
        [sys.executable, examples_path / "rotate_and_diff.py"],
        check=True,
        cwd=examples_path,
    )
