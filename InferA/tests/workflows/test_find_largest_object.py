import pytest
import pandas as pd
from unittest.mock import patch
from src.workflows.find_largest_object import FindLargestObject

@pytest.fixture
def workflow():
    return FindLargestObject(base_path="mock/path")

# ---- Test: run() with halo ----
@patch("src.workflows.find_largest_object.read_gio_to_df")
@patch("src.workflows.find_largest_object.pretty_print_df")
def test_run_halo_default_sort(mock_pretty, mock_read, workflow):
    mock_read.return_value = pd.DataFrame({"fof_halo_count": [10, 30, 20]})

    result = workflow.run(object_type="halo", timestep=1, n=2)

    assert result["fof_halo_count"].tolist() == [30, 20]
    mock_read.assert_called_once_with("mock/path", "halo", 1, "properties")
    mock_pretty.assert_called_once()

# ---- Test: run() with galaxy ----
@patch("src.workflows.find_largest_object.read_gio_to_df")
@patch("src.workflows.find_largest_object.pretty_print_df")
def test_run_galaxy_default_sort(mock_pretty, mock_read, workflow):
    mock_read.return_value = pd.DataFrame({"gal_count": [1, 9, 5]})

    result = workflow.run(object_type="galaxy", timestep=2, n=1)

    assert result["gal_count"].tolist() == [9]
    mock_read.assert_called_once_with("mock/path", "galaxy", 2, "properties")
    mock_pretty.assert_called_once()

# ---- Test: run() with invalid object_type ----
def test_run_invalid_object_type(workflow):
    with pytest.raises(ValueError, match="Unsupported object_type"):
        workflow.run(object_type="star", timestep=0, n=1)

# ---- Test: get_n_largest_objects() directly ----
@patch("src.workflows.find_largest_object.read_gio_to_df")
def test_get_n_largest_objects(mock_read, workflow):
    mock_read.return_value = pd.DataFrame({"mass": [4, 10, 2]})

    result = workflow.get_n_largest_objects("galaxy", 0, 2, sort_by="mass")

    assert result["mass"].tolist() == [10, 4]
    mock_read.assert_called_once_with("mock/path", "galaxy", 0, "properties")
