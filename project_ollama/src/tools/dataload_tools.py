import os
import sys
import re
import json
import logging

import pandas as pd
import numpy as np
import duckdb
from dsi.dsi import DSI

from typing import List, Tuple, Dict, Annotated
from collections import defaultdict
from tqdm import tqdm

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from src.core.state import State
from src.utils.config import WORKING_DIRECTORY
from src.utils.func_utils import time_function
from src.utils.dataframe_utils import pretty_print_df

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

logger = logging.getLogger(__name__)

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


@tool(parse_docstring=True)
def load_to_db(vars: list, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Load selected variables from multiple simulation files into a DuckDB database.
    This tool reads specific variables (`vars`) from multiple time-stepped simulation files,
    enriches them with metadata (simulation ID, time step, object), and writes the combined 
    data to a local DuckDB database file. Example: Call this tool to prepare and consolidate data for downstream querying.

    Args:
        vars: A list of variable names to extract from the data files.

    Returns:
        str: Path to the written database file for downstream querying.
    """
    file_index = state.get("file_index", None)
    session_id = state.get("session_id", None)

    if session_id:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}{session_id}.duckdb"
    else:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}data.duckdb"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(DUCKDB_DIRECTORY), exist_ok=True)
    # Write to DuckDB file
    con = duckdb.connect(DUCKDB_DIRECTORY)
    logger.info(f"[load_to_db() TOOL] Connected to database: {DUCKDB_DIRECTORY}")

    if not file_index:
        raise Exception("FileIndexError")
    table_initialized = False

    logger.info(f"[load_to_db() TOOL] Writing file index to db:")
    for sim in file_index:
        for ts in file_index[sim]:
            for obj, file_path in file_index[sim][ts].items():
                data = gio.read(file_path, vars)

                logger.info(f"      - File: {file_path}")
                df = pd.DataFrame(np.column_stack(data), columns=vars)

                # Add metadata
                df["simulation"] = int(sim)
                df["time_step"] = int(ts)
                df["object"] = str(obj)

                if not table_initialized:
                    con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM df")
                    table_initialized = True
                else:
                    con.execute("INSERT INTO data SELECT * FROM df")

    columns = con.table("data").columns
    df = con.sql("SELECT * FROM data").df()
    print("Complete data:")
    pretty_print_df(df, max_rows = 20)

    con.close()
    logger.info("[load_to_db() TOOL] Done writing to DB. Connection closed.")

    return Command(update={
        "db_path": DUCKDB_DIRECTORY,
        "messages": [
            ToolMessage(
                f"Wrote all data to {DUCKDB_DIRECTORY}. TABLE = 'data', Columns = {columns} All dataframes concatenated via: full_df = pd.concat(all_dfs, ignore_index=True)",
                tool_call_id=tool_call_id,
            )
        ]
    })


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
        result[i] = {}

        if timestep == [-1]:  # All timesteps
            for ts, data in index[sim_id].items():
                filtered = {obj: data[obj] for obj in object if obj in data}
                if filtered:
                    result[i][ts] = filtered
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
                    result[i][ts] = filtered
                    
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
            
            sim_id = sim_match.group()

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

# @time_function
# def load_db(data):
#     db = DSI(f"{WORKING_DIRECTORY}/124.duckdb", backend_name="DuckDB")
#     db.list()
#     db.display("data", 10)

def main():
    # load_file_index([0], [498], ["haloproperties"])
    load_to_db(["fof_halo_count", "fof_halo_tag", "fof_halo_mass"])
    db = duckdb.connect(f"{WORKING_DIRECTORY}/data.duckdb")
    db.sql("PRAGMA table_info('data')").show()
    sql_df = db.sql("SELECT fof_halo_tag, fof_halo_mass, fof_halo_count, time_step FROM data WHERE simulation = 0 AND time_step IN (498, 105) ORDER BY time_step, fof_halo_mass DESC LIMIT 5;").df()

if __name__ == "__main__":
    main()