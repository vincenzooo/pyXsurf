import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
script_dir = project_root / "docs" / "source" / "examples"
script_path = script_dir / "rotate_and_diff.py"


def main():
    """simply run the example script."""
    print("Hello from test_rotate_and_diff.py!")
    subprocess.run([sys.executable, script_path])


if __name__ == "__main__":
    main()
