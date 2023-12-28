import filecmp
from pathlib import Path

import pySurf.demo_module as psdm


def test_function():
    assert "pySurf function" == psdm.function()


def test_data_function():
    assert "pySurf functionality data\n" == psdm.data_function()


def test_common_data_function():
    assert "common functionality data\n" == psdm.common_data_function()


def test_test_data(test_path: Path):
    with open(test_path / "data" / "test_data.txt", "r") as f:
        assert "test data\n" == f.read()


def test_image_generate_file(test_path, tmp_path):
    # Prepare output directory
    output_tmp_dir: Path = tmp_path / "units.pySurf.test_demo_module"
    output_tmp_dir.mkdir(parents=True, exist_ok=True)
    # Run function which generates file
    psdm.image_generate_file(output_tmp_dir / "test_image.png")

    # Compare generated file with golden image
    input_test_dir: Path = test_path / "units" / "pySurf" / "data"
    assert filecmp.cmp(
        input_test_dir / "test_golden_image.png",
        output_tmp_dir / "test_image.png",
        shallow=False,
    )
