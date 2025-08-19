"""
Module: data_loader.py
Purpose: Provides tools and functions to:
    - Build an index of simulation data files.
    - Load selected simulation data into a DuckDB database.
    - Enrich the data with metadata such as timestep and object type.
    - Integrate with LangGraph via @tool decorators for structured pipelines.

Main Tools:
    - load_file_index: Indexes file paths for selected simulations, timesteps, and object types.
    - load_to_db: Loads selected columns from simulation files into a DuckDB table with metadata.
    - index_simulation_directories: Helper to recursively index the simulation analysis folders.
"""


import os
import sys
import re
import json
import logging

import pandas as pd
import numpy as np
import duckdb

from typing import List, Tuple, Dict, Annotated
from collections import defaultdict
from tqdm import tqdm

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from src.utils.dataframe_utils import pretty_print_df
from config import WORKING_DIRECTORY, SIMULATION_PATHS

# Add genericio to sys path
genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

logger = logging.getLogger(__name__)



@tool(parse_docstring=True)
def load_to_db(columns: list, object_type: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    This tool reads specific columns from multiple time-stepped simulation files for only one object, and
    enriches it with metadata (simulation ID, time step, object, x, y, z-coordinates), and writes the data to a local DuckDB database file.

    Args:
        columns: A list of column names to extract from the data files.
        object_type: A single object_type being loaded to the database.

    Returns:
        str: Path to the written database file for downstream querying.
    """
    file_index = state.get("file_index", None)
    session_id = state.get("session_id", None)
    state_key = state.get("state_key", "")

    db_columns = state.get("db_columns", [])
    db_tables = state.get("db_tables", [])

    if not isinstance(columns, list) or not columns:
        raise ValueError("MissingColumnListError: You must provide a non-empty list of columns to extract.")
    if not file_index:
        raise ValueError("MissingFileIndexError: The file index is missing from state. Use the DataLoader to extract relevant files.")

    # Construct database path
    if session_id:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}{session_id}/{session_id}.duckdb"
    else:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}{session_id}/data.duckdb"

    # Write to DuckDB file
    try:
        con = duckdb.connect(DUCKDB_DIRECTORY)
    except Exception as e:
        raise RuntimeError(f"DatabaseConnectionError: Could not connect to DuckDB at {DUCKDB_DIRECTORY}. Error: {e}")

    logger.debug(f"[load_to_db() TOOL] Connected to database: {DUCKDB_DIRECTORY}")
    table_initialized = False

    # Gather all files matching object type
    all_files = [
        (sim, ts, obj, file_path)
        for sim in file_index
        for ts in file_index[sim]
        for obj, file_path in file_index[sim][ts].items()
        if obj == object_type
    ]  

    total_files = sum(len(file_index[sim][ts]) for sim in file_index for ts in file_index[sim])
    iterator = tqdm(all_files, desc= "Loading files") if total_files > 10 else all_files
    logger.debug(f"[load_to_db() TOOL] Writing file index to db. Total files to write: {total_files}\n       Using columns: {columns}")

    for sim, ts, obj, file_path in iterator:
        try:
            data = gio.read(file_path, columns)
        except Exception as e:
            raise ValueError(
                f"ColumnReadError: Failed to read columns {columns} from {file_path}. \nLikely cause: missing or misspelled column names. \n    Error: {e}"
            )
        
        if total_files <= 5:
            logger.info(f"\n- File: {file_path}")

        df = pd.DataFrame(np.column_stack(data), columns=columns)
        df["simulation"] = int(sim)
        df["time_step"] = int(ts)
        df["object_type"] = str(obj)

        try:
            if not table_initialized:
                con.execute(f"CREATE OR REPLACE TABLE {object_type} AS SELECT * FROM df")
                table_initialized = True
            else:
                con.execute(f"INSERT INTO {object_type} SELECT * FROM df")
        except Exception as e:
            raise RuntimeError(f"DatabaseWriteError: Failed writing to DuckDB from file {file_path}. Error: {e}")

    try:
        columns = con.table(f"{object_type}").columns
        df = con.sql(f"SELECT * FROM {object_type} LIMIT 5").df()
        pretty_print_df(df, max_rows = 5)
    except Exception as e:
        raise RuntimeError(f"DatabaseReadError: Failed to finalize or read from DuckDB. Error: {e}")
    finally:
        con.close()

    logger.debug("[load_to_db() TOOL] Done writing to DB. Connection closed.")

    db_columns.append(columns)
    db_tables.append(object_type)

    return Command(update={
        "db_path": DUCKDB_DIRECTORY,
        "db_tables": db_tables,
        "db_columns" : db_columns,
        "messages": [
            ToolMessage(
                f"Wrote all data to {DUCKDB_DIRECTORY}. TABLE = '{object_type}', Columns = {columns} All dataframes concatenated via: full_df = pd.concat(all_dfs, ignore_index=True)",
                tool_call_id=tool_call_id,
            )
        ]
    })


@tool(parse_docstring=True)
def load_file_index(sim_idx: list, timestep: list, object: list, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Load file index from a simulation file based on simulation index, timestep and object to load. 'simulation', 'timestep' and 'object' must be explicitly stated.
    
    Args:
        sim_idx: List of simulation indexes from the simulation path. Index starts at 0. If user asks for all simulations or is unclear which simulation to use, instead load all simulations - set simulation = [-1]
        timestep: List of timestep(s) of simulation to load from. If user asks for all timesteps or is unclear which timestep to use, instead load all timesteps - set timestep = [-1].
        object: List of object types in simulation to load (maximum of 4 object types). Must be one of the following: [haloproperties, galaxyproperties, haloparticles, galaxyparticles]

    Returns:
        dict: Nested dictionary index of all relevant files
    """
    # Show available simulation paths
    for idx, path in enumerate(SIMULATION_PATHS):
        logger.info(f"Simulation {idx}: {path}")

    with open("src/data/file_descriptions.json", "r") as file:
        valid_object_types = json.load(file).keys()

    index = index_simulation_directories(SIMULATION_PATHS, valid_object_types)
    sim_ids = list(index.keys())

    # Normalize inputs
    sim_idx = [sim_idx] if isinstance(sim_idx, int) else sim_idx
    timestep = [timestep] if isinstance(timestep, int) else timestep
    object = [object] if isinstance(object, str) else object

    # sim_idx = [-1] means "all simulations"
    if sim_idx == [-1]:
        sim_idx = list(range(len(sim_ids)))  # Assumes 1-to-1 with SIMULATION_PATHS

    result = {}
    
    for i in sim_idx:
        sim_id = sim_ids[i]
        result[i] = {}

        if timestep == [-1]:
            # Include all timesteps
            for ts, data in index[sim_id].items():
                filtered = {obj: data[obj] for obj in object if obj in data}
                if filtered:
                    result[i][ts] = filtered
        else:  # Specific timesteps
            # Validate requested timesteps
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
        "object_type": object,
        "messages": [
            ToolMessage(
                "Successfully loaded file index information",
                tool_call_id=tool_call_id,
            )
        ]
    })


def index_simulation_directories(root_paths: List[str], valid_object_types: set) -> Dict[Tuple[float, float, float], Dict[int, Dict[str, str]]]:
    """
    Recursively indexes all simulation files in analysis folders.

    Expected structure:
        ROOT/FSN_{float}_VEL_{float}_TEXP_{float}/analysis/m000p-{timestep}.{object_type}

    Args:
        root_paths (List[str]): Base simulation directories.
        valid_object_types (set): Valid file suffixes (object types) to include.

    Returns:
        Dict: A nested dict with the structure:
              {simulation_id: {timestep: {object_type: file_path}}}
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



# Optional for testing or CLI
if __name__ == "__main__":
    raise NotImplementedError("This script is intended to be used as a module, not run directly. You can use this to test your data loading function.")