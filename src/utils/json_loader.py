"""
Module: json_utils.py
Purpose: Utility functions for working with the structured data variable JSON file,
         including loading the file, extracting variable names, and retrieving descriptions.

Functions:
    - open_json(filepath): Loads JSON from a file.
    - get_variable_names_from_json(object_type): Returns all variable names for a given object type.
    - get_field_descriptions_from_json(object_type): Returns field:description mapping for an object type.
"""

import json
from config import DATA_DICT_PATH

# Path to the structured variable description file
DATA_VARIABLES = f"{DATA_DICT_PATH}data_variables.json"


def open_json(filepath: str):
    """Opens and reads a JSON file from the given path."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def get_variable_names_from_json(object_type: str) -> dict:
    """
    Extract variable names for a given object from a nested JSON structure.

    Args:
        data (dict): The full JSON data.
        object_type (str): The top-level key whose variables you want to extract.

    Returns:
        dict: A dictionary of the form {object_type: [var1, var2, ...]}
    """
    data = open_json(DATA_VARIABLES)
    if object_type not in data:
        raise ValueError(f"Object '{object_type}' not found in data.")

    variables = list(data[object_type]["columns"].keys())
    return variables


def get_field_descriptions_from_json(object_type: str) -> dict:
    """
    Extract a dictionary of field: description pairs for a given object type from a nested JSON structure.

    Args:
        object_type (str): The top-level key (e.g. 'accumulatedcores') whose fields you want to extract.
        data (dict): The full JSON data structure.

    Returns:
        dict: A dictionary in the form {field_name: description}
    """
    data = open_json(DATA_VARIABLES)

    if object_type not in data:
        raise ValueError(f"Object '{object_type}' not found in data.")

    field_descriptions = {}
    for field, info in data[object_type]["columns"].items():
        description = info.get("description", "").strip()
        if description.startswith("-"):
            description = description.lstrip("- ").strip()
        field_descriptions[field] = description

    return field_descriptions