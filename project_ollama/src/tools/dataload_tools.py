import os
import re
import json
import logging
from typing import List, Tuple, Dict, Annotated
from collections import defaultdict
from tqdm import tqdm

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

from src.utils.config import WORKING_DIRECTORY
from src.utils.genericio_utils import load_gio_to_df

# Ensure the working directory exists
if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
    logger.info(f"Created working directory: {WORKING_DIRECTORY}")


def normalize_path(file_path: str) -> str:
    """
    Normalize file path for cross-platform compatibility.
    
    Args:
    file_path (str): The file path to normalize
    
    Returns:
    str: Normalized file path
    """
    if WORKING_DIRECTORY not in file_path:
        file_path = os.path.join(WORKING_DIRECTORY, file_path)
    return os.path.normpath(file_path)

# @tool
# def write_index_to_db(file_index: dict, db_path: Annotated[str, "Path to the db file"] = './db.duckdb') -> str:
#     """
#     Load all data files from the full file index
    
#     Args:
#         index: Nested dictionary structured as index[sim_id][timestep][object] = path or list of paths.
    
#     Returns:
#         dict: A dictionary of data frames organized by data[sim_id][timestep][object] = pd.dataframe
#     """
#     try:
#         file_path = normalize_path(db_path)
#         logger.info(f"[DATA LOAD TOOL] Creating db: {file_path}")

#         data = {}
#         for sim_id, ts_dict in file_index.items():
#             for ts, object_dict in tqdm(ts_dict.items()):
#                 for object, file_path in object_dict.items():
#                     data[sim_id][ts][object] = load_gio_to_df(file_path)
#         return data
    
#     except Exception as e:
#         logger.error(f"Error while saving db: {str(e)}")
#         return f"Error while saving db: {str(e)}"


@tool(parse_docstring=True)
def load_file_index(sim_idx: list, timestep: list, object: list, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Load file index from a simulation file based on simulation index, timestep and object to load. 'simulation', 'timestep' and 'object' must be explicitly stated.
    
    Args:
        sim_idx: List of simulation indexes. Index can be between 0-3.
        timestep: List of timestep(s) of simulation to load from. If user asks for all timesteps, set timestep = [-1]
        object: List of object types in simulation to load (maximum of 4 object types). Must be one of the following: [haloproperties, galaxyproperties, haloparticles, galaxyparticles]

    Returns:
        dict: Nested dictionary index of all relevant files
    """
    root_paths = ["/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B", "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3A"]
    
    with open("src/data/file_descriptions.json", "r") as file:
        valid_object_types = json.load(file).keys()

    index = index_simulation_directories(root_paths, valid_object_types)
    sim_ids = list(index.keys())

    if isinstance(sim_idx, int):
        sim_idx = [sim_idx]

    if isinstance(timestep, int):
        timestep = [timestep]

    if isinstance(object, str):
        object = [object]

    result = {}
    
    for i in sim_idx:
        sim_id = sim_ids[i]
        result[sim_id] = {}

        if timestep == [-1]:  # All timesteps
            for ts, data in index[sim_id].items():
                filtered = {obj: data[obj] for obj in object if obj in data}
                if filtered:
                    result[sim_id][ts] = filtered
        else:  # Specific timesteps
            # Validate all timesteps first
            missing_timesteps = [ts for ts in timestep if ts not in index[sim_id]]
            if missing_timesteps:
                raise Exception("TimeStepError")

            # Load requested timesteps
            for ts in timestep:
                data_at_ts = index[sim_id][ts]
                filtered = {obj: data_at_ts[obj] for obj in object if obj in data_at_ts}
                if filtered:
                    result[sim_id][ts] = filtered
                    
    return Command(update={
        "file_index": result,
        "messages": [
            ToolMessage(
                "Successfully loaded file index information",
                tool_call_id=tool_call_id,
            )
        ]
    })


def index_simulation_directories(root_paths: List[str], valid_object_types: set) -> Dict[Tuple[float, float, float], Dict[int, Dict[str, str]]]:
    """
    Builds a hierarchical index from simulation analysis directories.
    Expected file structure:
    ROOT/FSN_{float}_VEL_{float}_TEXP_{float}/analysis/m000p-{timestep}.{object_type}

    Only object types in `valid_object_types` are included.
    """
    index = defaultdict(lambda: defaultdict(dict))

    float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    sim_pattern = re.compile(
        rf"FSN_{float_pattern}_VEL_{float_pattern}_TEXP_{float_pattern}_BETA_{float_pattern}_SEED_{float_pattern}"
    )
    file_pattern = re.compile(r"m000p-(\d+)\.(.+)$")

    for root_path in root_paths:
        for dirpath, _, filenames in os.walk(root_path):
            if os.path.basename(dirpath) != "analysis":
                continue

            sim_match = sim_pattern.search(dirpath)
            if not sim_match:
                continue

            sim_id = tuple(map(float, sim_match.groups()))

            for fname in filenames:
                file_match = file_pattern.match(fname)
                if not file_match:
                    continue

                timestep = int(file_match.group(1))
                object_type = file_match.group(2)

                if object_type not in valid_object_types:
                    continue

                full_path = os.path.join(dirpath, fname)
                index[sim_id][timestep][object_type] = full_path

    return index


# def main():
#     index = load_file_index([0], [102], ["haloproperties"], 1)
#     # write_index_to_db(index)

# if __name__ == "__main__":
#     main()