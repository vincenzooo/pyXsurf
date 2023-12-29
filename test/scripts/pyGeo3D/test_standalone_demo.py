import subprocess
import sys


def test_main():
    """run a subprocess to execute the "pyGeo3D.standalone_demo" module."""
    subprocess.run(
        [sys.executable, "-m", "pyGeo3D.standalone_demo"],
        check=True,
    )
