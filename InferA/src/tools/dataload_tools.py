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


@tool(parse_docstring=True)
def load_to_db(columns: list, object_type: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    This tool reads specific columns from multiple time-stepped simulation files for only one object,
    enriches it with metadata (simulation ID, time step, object), and writes the data to a local DuckDB database file. 
    Example: If given the task "Load halo data", run with columns for haloproperties (including fof_halo_tag the unique identifier).

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

    if session_id:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}{state_key}/{session_id}.duckdb"
    else:
        DUCKDB_DIRECTORY = f"{WORKING_DIRECTORY}{state_key}/data.duckdb"

    # Write to DuckDB file
    try:
        con = duckdb.connect(DUCKDB_DIRECTORY)
    except Exception as e:
        raise RuntimeError(f"DatabaseConnectionError: Could not connect to DuckDB at {DUCKDB_DIRECTORY}. Error: {e}")

    logger.debug(f"[load_to_db() TOOL] Connected to database: {DUCKDB_DIRECTORY}")
    table_initialized = False
    
    total_files = sum(len(file_index[sim][ts]) for sim in file_index for ts in file_index[sim])
    logger.debug(f"[load_to_db() TOOL] Writing file index to db. Total files to write: {total_files}\n       Using columns: {columns}")

    all_files = [
        (sim, ts, obj, file_path)
        for sim in file_index
        for ts in file_index[sim]
        for obj, file_path in file_index[sim][ts].items()
        if obj == object_type
    ]

    iterator = tqdm(all_files, desc= "Loading files") if total_files > 10 else all_files

    for sim, ts, obj, file_path in iterator:
        try:
            data = gio.read(file_path, columns)
        except Exception as e:
            raise ValueError(
                f"ColumnReadError: Failed to read columns {columns} from {file_path}. Likely cause: missing or misspelled column names. Error: {e}"
            )
        
        if total_files <= 5:
            logger.info(f"\n- File: {file_path}")

        df = pd.DataFrame(np.column_stack(data), columns=columns)

        # Add metadata
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
        df = con.sql(f"SELECT * FROM {object_type}").df()
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
        sim_idx: List of simulation indexes. Index can be between 0-3.
        timestep: List of timestep(s) of simulation to load from. If user asks for all timesteps, set timestep = [-1]
        object: List of object types in simulation to load (maximum of 4 object types). Must be one of the following: [haloproperties, galaxyproperties, haloparticles, galaxyparticles]

    Returns:
        dict: Nested dictionary index of all relevant files
    """
    root_paths = ["/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B", "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3A", "/vast/projects/libra/scidac_data/128MPC_RUNS_FLAMNGO_DESIGN_3A", "/vast/projects/libra/scidac_data/128MPC_RUNS_FLAMNGO_DESIGN_3B"]
    for idx, path in enumerate(root_paths):
        logger.info(f"Simulation {idx}: {path}")

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
    # db = duckdb.connect(f"{WORKING_DIRECTORY}/data.duckdb")
    # db.sql("PRAGMA table_info('data')").show()
    # sql_df = db.sql("SELECT fof_halo_tag, fof_halo_mass, fof_halo_count, time_step FROM data WHERE simulation = 0 AND time_step IN (498, 105) ORDER BY time_step, fof_halo_mass DESC LIMIT 5;").df()

if __name__ == "__main__":
    main()