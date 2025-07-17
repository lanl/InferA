import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from InferA.src.old_nodes.visualize_object import VisualizeObject

@pytest.fixture
def workflow():
    return VisualizeObject(base_path="mock/base_path", full_path="mock/full_path")

@pytest.fixture
def sample_df_halo():
    return pd.DataFrame({
        "fof_halo_center_x": [0, 10, 20],
        "fof_halo_center_y": [0, 10, 20],
        "fof_halo_center_z": [0, 10, 20],
        "sod_halo_radius": [5, 5, 5],
    })

@pytest.fixture
def sample_df_galaxy():
    return pd.DataFrame({
        "gal_center_x": [0, 15, 30],
        "gal_center_y": [0, 15, 30],
        "gal_center_z": [0, 15, 30],
        "gal_radius": [3, 3, 3],
    })

# ---- Test: run() with halo ----
@patch("src.workflows.visualize_object.VisualizeObject.render_region_only")
@patch("src.workflows.visualize_object.VisualizeObject.plot")
def test_run_halo(mock_plot, mock_render, workflow, sample_df_halo):
    mock_render.return_value = np.array([[0,0,0], [1,1,1]])
    mock_plot.return_value = "output/plot.vtp"

    result = workflow.run(sample_df_halo, object_type="halo", timestep=1, output_name="plot")

    assert result == "output/plot.vtp"
    mock_render.assert_called_once()
    mock_plot.assert_called_once()

# ---- Test: run() with galaxy ----
@patch("src.workflows.visualize_object.VisualizeObject.render_region_only")
@patch("src.workflows.visualize_object.VisualizeObject.plot")
def test_run_galaxy(mock_plot, mock_render, workflow, sample_df_galaxy):
    mock_render.return_value = np.array([[0,0,0], [1,1,1]])
    mock_plot.return_value = "output/plot.vtp"

    result = workflow.run(sample_df_galaxy, object_type="galaxy", timestep=2, output_name="plot")

    assert result == "output/plot.vtp"
    mock_render.assert_called_once()
    mock_plot.assert_called_once()

# ---- Test: run() with invalid object_type ----
def test_run_invalid_object_type(workflow, sample_df_halo):
    with pytest.raises(ValueError, match="Unsupported object_type"):
        workflow.run(sample_df_halo, object_type="star", timestep=0)

# ---- Test: plot() output filename endings ----
@patch("src.workflows.visualize_object.pv.Plotter")
def test_plot_file_extension(mock_plotter, workflow):
    points_bg = np.array([[0, 0, 0]])
    points_target = np.array([[1, 1, 1]])
    center = [0, 0, 0]
    radius = 25

    mock_instance = MagicMock()
    mock_plotter.return_value = mock_instance

    filename_vtp = workflow.plot(points_bg, points_target, center, radius, "testfile", save_as_vtp=True)
    assert filename_vtp.endswith(".vtp")

    filename_html = workflow.plot(points_bg, points_target, center, radius, "testfile", save_as_vtp=False)
    assert filename_html.endswith(".html")
