import pytest
import json
import tempfile
from unittest.mock import patch, mock_open

import src.utils.json_loader as jl


def test_open_json_reads_file():
    sample_data = {"test": 123}
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        json.dump(sample_data, tf)
        tf.seek(0)
        tf.flush()
        data = jl.open_json(tf.name)
    assert data == sample_data


@patch("src.utils.json_loader.open_json")
def test_get_variable_names_from_json_valid(mock_open_json):
    mock_open_json.return_value = {
        "halo": {
            "x": {"description": "X position"},
            "y": {"description": "Y position"},
            "z": {"description": "Z position"}
        }
    }
    result = jl.get_variable_names_from_json("halo")
    assert set(result) == {"x", "y", "z"}


@patch("src.utils.json_loader.open_json")
def test_get_variable_names_from_json_invalid_key(mock_open_json):
    mock_open_json.return_value = {"other": {}}
    with pytest.raises(ValueError, match="Object 'halo' not found in data."):
        jl.get_variable_names_from_json("halo")


@patch("src.utils.json_loader.open_json")
def test_get_field_descriptions_from_json_valid(mock_open_json):
    mock_open_json.return_value = {
        "halo": {
            "x": {"description": "X pos"},
            "y": {"description": "- Y pos"},
            "z": {"description": "   - Z pos   "}
        }
    }
    result = jl.get_field_descriptions_from_json("halo")
    assert result == {
        "x": "X pos",
        "y": "Y pos",
        "z": "Z pos"
    }


@patch("src.utils.json_loader.open_json")
def test_get_field_descriptions_from_json_invalid_key(mock_open_json):
    mock_open_json.return_value = {"star": {}}
    with pytest.raises(ValueError, match="Object 'halo' not found in data."):
        jl.get_field_descriptions_from_json("halo")


def test_extract_code_block_found():
    text = "Here is code:\n```python\nprint('hello')\n```"
    result = jl.extract_code_block(text)
    assert result == "python\nprint('hello')"


def test_extract_code_block_not_found():
    text = "No code block here."
    result = jl.extract_code_block(text)
    assert result == "No code block here."
