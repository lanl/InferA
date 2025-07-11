import json
import re


def open_json(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

DATA_VARIABLES = "src/data/JSON/data_variables.json"

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


def extract_code_block(text):
    """
    Extracts the full code block (including the language identifier, like ```python).
    Does not strip out the 'python' or other language hint.
    """
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()