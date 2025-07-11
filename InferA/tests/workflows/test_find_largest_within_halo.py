import pytest
import pandas as pd
from unittest.mock import patch
from src.workflows.find_largest_within_halo import FindLargestWithinHalo

@pytest.fixture
def workflow():
    return FindLargestWithinHalo(base_path="mock/path")

# ---- Test: run() with galaxy (default sort_by) ----
@patch("src.workflows.find_largest_within_halo.read_gio_to_df")
@patch("src.workflows.find_largest_within_halo.pretty_print_df")
def test_run_galaxy_default_sort(mock_pretty, mock_read, workflow):
    # Mock DataFrame including 'fof_halo_tag' column
    mock_read.return_value = pd.DataFrame({
        "gal_count": [10, 50, 20],
        "fof_halo_tag": [123, 123, 999]
    })

    result = workflow.run(object_type="galaxy", timestep=1, halo_id=123, n=2)

    # Should filter only rows with fof_halo_tag == 123 and sort descending by gal_count
    expected_counts = [50, 10]
    assert result["gal_count"].tolist() == expected_counts
    mock_read.assert_called_once_with("mock/path", "galaxy", 1, "properties")
    mock_pretty.assert_called_once()

# ---- Test: run() with invalid object_type ----
def test_run_invalid_object_type(workflow):
    with pytest.raises(ValueError, match="Unsupported object_type"):
        workflow.run(object_type="halo", timestep=0, halo_id=123, n=1)

# ---- Test: get_n_objects_in_timestep_halo() directly ----
@patch("src.workflows.find_largest_within_halo.read_gio_to_df")
def test_get_n_objects_in_timestep_halo(mock_read, workflow):
    df = pd.DataFrame({
        "gal_count": [5, 8, 2, 9],
        "fof_halo_tag": [1, 2, 1, 1]
    })
    mock_read.return_value = df

    result = workflow.get_n_objects_in_timestep_halo("galaxy", 0, halo=1, n=2)

    # Should filter to fof_halo_tag == 1, then get top 2 gal_count (9,5)
    expected = [9, 5]
    assert result["gal_count"].tolist() == expected
    mock_read.assert_called_once_with("mock/path", "galaxy", 0, "properties")
