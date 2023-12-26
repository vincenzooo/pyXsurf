import subprocess
import sys


def main():
    """run a subprocess to execute the "pyGeo3D.standalone_demo" module."""
    print("Hello from test_standalone_demo.py!")
    subprocess.run([sys.executable, "-m", "pyGeo3D.standalone_demo"])


if __name__ == "__main__":
    main()
