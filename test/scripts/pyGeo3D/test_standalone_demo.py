import subprocess
import sys


def main():
    print("Hello from test_standalone_demo.py!")
    subprocess.run([sys.executable, "-m", "pyXsurf.pyGeo3D.standalone_demo"])


if __name__ == "__main__":
    main()
