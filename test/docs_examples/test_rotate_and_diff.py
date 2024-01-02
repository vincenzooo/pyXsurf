import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path


def test_main(examples_path: Path, tmp_path: Path):
    """Run system Python interpreter on the test script corresponding to this test."""
    # Use this if we want to override the input files
    # input_dir: Path = examples_path / "inputs"
    expected_output_dir: Path = examples_path / "outputs" / "rotate_and_diff"
    output_tmp_dir: Path = tmp_path / "outputs" / "rotate_and_diff"

    subprocess.run(
        [
            sys.executable,
            examples_path / "rotate_and_diff.py",
            # f"--indir={input_dir}",
            f"--outdir={output_tmp_dir}",
        ],
        check=True,
        cwd=examples_path,
    )

    for f in ["alignment.png", "difference.png", "difference.dat"]:
        pass
        # assert filecmp.cmp(
        #     expected_output_dir / f,
        #     output_tmp_dir / f,
        #     shallow=False,
        # )
