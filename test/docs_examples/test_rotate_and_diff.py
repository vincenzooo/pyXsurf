import subprocess

"""
main runs a Python .py script. script path is hardcoded as relative path from from this file.

> python test_rotate_and_diff.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent  # root relative to this file
script_dir = (
    project_root / "docs" / "source" / "examples"
)  # location of script to run wrt root
script_path = script_dir / "rotate_and_diff.py"


def main():
    """Run system Python interpreter on the test script corresponding to this test."""
    print("Hello from test_rotate_and_diff.py!")
    subprocess.run([sys.executable, script_path])


if __name__ == "__main__":
    main()
