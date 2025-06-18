import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.utils import genericio_utils as gio_utils


def test_list_genericio_files(tmp_path):
    # Create some test files
    (tmp_path / "file1.txt").write_text("data")
    (tmp_path / "file#skip.txt").write_text("data")

    files = gio_utils.list_genericio_files(str(tmp_path))
    assert "file1.txt" in files
    assert "file#skip.txt" not in files

    with pytest.raises(ValueError):
        gio_utils.list_genericio_files("not/a/real/path")


def test_subselect_points_valid():
    arr1 = np.arange(100)
    arr2 = np.arange(100, 200)
    result, size = gio_utils.subselect_points([arr1, arr2], sample_fraction=0.2)
    assert len(result) == 2
    assert result[0].shape[0] == size
    assert result[1].shape[0] == size


def test_subselect_points_invalid():
    with pytest.raises(ValueError):
        gio_utils.subselect_points([])

    arr1 = np.arange(10)
    arr2 = np.arange(11)
    with pytest.raises(ValueError):
        gio_utils.subselect_points([arr1, arr2])


def test_box_box_intersect():
    b1 = np.array([0, 0, 0, 5, 5, 5])
    b2 = np.array([4, 4, 4, 6, 6, 6])
    assert gio_utils.box_box_intersect(b1, b2)

    b3 = np.array([6, 6, 6, 7, 7, 7])
    assert not gio_utils.box_box_intersect(b1, b3)


def test_check_fully_inside():
    outer = np.array([0, 0, 0, 10, 10, 10])
    inner = np.array([2, 2, 2, 5, 5, 5])
    assert gio_utils.check_fully_inside(inner, outer)
    assert not gio_utils.check_fully_inside(outer, inner)


def test_create_pos_BB():
    pos = [5, 5, 5]
    dims = [1, 2, 3]
    bb = gio_utils.create_pos_BB(pos, dims)
    assert np.allclose(bb, [4, 3, 2, 6, 7, 8])


def test_compute_intersection():
    pos = [1, 1, 1]
    bb_list = [
        np.array([0, 0, 0, 2, 2, 2]),
        np.array([10, 10, 10, 12, 12, 12])
    ]
    result = gio_utils.compute_intersection(pos, bb_list, 1, 1, 1)
    assert result == [0]


@patch("src.utils.genericio_utils.gio.read")
def test_region_filter(mock_read):
    mock_read.return_value = [
        np.array([1, 2, 3]),
        np.array([10, 20, 30])
    ]
    result = gio_utils.region_filter("dummy_file", ["x", "y"], 0, 2)
    assert len(result) == 2
    assert result[0][0] == 2


@patch("src.utils.genericio_utils.gio.read")
def test_read_output_file(mock_read):
    mock_read.return_value = [np.array([1]), np.array([2]), np.array([3])]
    result = gio_utils.read_output_file("/tmp", 42)
    assert len(result) == 3
    mock_read.assert_called_once()


@patch("src.utils.genericio_utils.gio.read")
@patch("src.utils.genericio_utils.json_loader.get_variable_names_from_json")
def test_read_gio_to_df(mock_get_vars, mock_read):
    mock_get_vars.return_value = ["x", "y"]
    mock_read.return_value = [np.array([1.0]), np.array([2.0])]
    df = gio_utils.read_gio_to_df("/tmp", "halo", 100, ".dat")
    assert "x" in df.columns
    assert "y" in df.columns
