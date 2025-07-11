import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.workflows.track_object_evolution import TrackObjectEvolution

@pytest.fixture
def workflow():
    return TrackObjectEvolution(base_path="mock/path")

# Mock tqdm to just return iterable (disable progress bar)
@pytest.fixture(autouse=True)
def patch_tqdm():
    with patch("src.workflows.track_object_evolution.tqdm", side_effect=lambda x, **kwargs: x):
        yield

# Helper to create a mock DataFrame for halos/galaxies
def make_mock_df():
    return pd.DataFrame({
        "fof_halo_tag": ["halo1", "halo2"],
        "fof_halo_center_x": [0.0, 1.0],
        "fof_halo_center_y": [0.0, 1.0],
        "fof_halo_center_z": [0.0, 1.0],
        "fof_halo_count": [10, 9],
        "sod_halo_radius": [100, 110]
    })

# --- run() basic success for halo ---
@patch("src.workflows.track_object_evolution.list_genericio_files")
@patch("src.workflows.track_object_evolution.read_gio_to_df")
def test_run_success(mock_read, mock_list, workflow):
    mock_list.return_value = [
        "m000p-1.halopproperties", 
        "m000p-2.halopproperties",
        "m000p-3.halopproperties",
    ]
    # Timestep keys extracted from filenames
    timesteps = [1, 2, 3]

    # The object to track exists in timestep 2
    df = make_mock_df()
    mock_read.return_value = df

    # Patch timesteps inside run: must return integer timesteps matching above
    # But also the file suffix is ".halopproperties" not ".halo" + "properties", fix the file suffix or the mock filenames to match your code.
    # To avoid mismatch, let's patch list_genericio_files to return expected pattern for your method:
    mock_list.return_value = [
        "m000p-1.haloproperties", 
        "m000p-2.haloproperties",
        "m000p-3.haloproperties",
    ]

    # Call run, object_id matches the first halo tag in mock df
    list_of_df, compiled_df = workflow.run(object_type="halo", object_id="halo1", timestep=2)

    # Result is list of dataframes for each matched timestep
    assert isinstance(list_of_df, list)
    assert all(isinstance(df, pd.DataFrame) for df in list_of_df)
    assert any("fof_halo_tag" in df.columns for df in list_of_df)

    assert isinstance(compiled_df, pd.DataFrame)
    assert "fof_halo_tag" in compiled_df

# --- run() timestep not found error ---
@patch("src.workflows.track_object_evolution.list_genericio_files")
def test_run_timestep_not_found(mock_list, workflow):
    mock_list.return_value = ["m000p-1.haloproperties"]
    with pytest.raises(ValueError, match="Timestep 99 not found"):
        workflow.run(object_type="halo", object_id="halo1", timestep=99)

# --- run() object_id not found error ---
@patch("src.workflows.track_object_evolution.list_genericio_files")
@patch("src.workflows.track_object_evolution.read_gio_to_df")
def test_run_object_id_not_found(mock_read, mock_list, workflow):
    mock_list.return_value = ["m000p-1.haloproperties"]
    # Return df with no matching object_id
    mock_read.return_value = pd.DataFrame({
        "fof_halo_tag": ["other"],
        "fof_halo_center_x": [0],
        "fof_halo_center_y": [0],
        "fof_halo_center_z": [0],
        "fof_halo_count": [5],
        "sod_halo_radius": [10]
    })
    with pytest.raises(ValueError, match="Halo ID halo1 not found"):
        workflow.run(object_type="halo", object_id="halo1", timestep=1)

# --- track_direction() basic matching test ---
@patch("src.workflows.track_object_evolution.read_gio_to_df")
def test_track_direction_basic_match(mock_read, workflow):
    # Provide two timesteps: 2 and 3
    timesteps = [2, 3]

    # Mock DataFrames for timesteps:
    # First timestep: coords close to start_center, count close enough -> match
    df1 = pd.DataFrame({
        "fof_halo_tag": ["halo1", "halo2"],
        "fof_halo_center_x": [0.1, 5],
        "fof_halo_center_y": [0.1, 5],
        "fof_halo_center_z": [0.1, 5],
        "fof_halo_count": [10, 20],
        "sod_halo_radius": [100, 200]
    })
    # Second timestep: no matches (too far)
    df2 = pd.DataFrame({
        "fof_halo_tag": ["halo3"],
        "fof_halo_center_x": [100],
        "fof_halo_center_y": [100],
        "fof_halo_center_z": [100],
        "fof_halo_count": [9],
        "sod_halo_radius": [300]
    })

    # Setup side effects of read_gio_to_df for timestep 2 and 3
    mock_read.side_effect = [df1, df2]

    results = workflow.track_direction(
        timesteps=timesteps,
        object_type="halo",
        file_type="properties",
        start_center=np.array([0, 0, 0]),
        start_count=10,
        distance_threshold=1.5,
        count_tolerance=0.2
    )

    # We expect 1 match from timestep 2, then stop at timestep 3 (no match)
    assert len(results) == 1
    assert results[0]["fof_halo_tag"] == "halo1"

# --- track_direction() exception handling test ---
@patch("src.workflows.track_object_evolution.read_gio_to_df")
def test_track_direction_exception_handling(mock_read, workflow):
    # Cause read_gio_to_df to throw an exception on first call
    def side_effect(*args, **kwargs):
        raise RuntimeError("Mock error")

    mock_read.side_effect = side_effect

    results = workflow.track_direction(
        timesteps=[1],
        object_type="halo",
        file_type="properties",
        start_center=np.array([0, 0, 0]),
        start_count=10,
        distance_threshold=1.5,
        count_tolerance=0.2
    )
    # Should skip timestep and return empty list, not raise
    assert results == []
