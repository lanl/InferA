import sys
import logging
import pandas as pd
import numpy as np
import duckdb
from typing import Annotated
from tqdm import tqdm

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from src.utils.config import WORKING_DIRECTORY
from src.utils.dataframe_utils import pretty_print_df

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

logger = logging.getLogger(__name__)

@tool(parse_docstring=True)
def track_halo_evolution(halo_id: str, timestep: int, x_column_name: str, y_column_name: str, z_column_name: str, size_column_name: str, id_column_name: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    This tool takes one halo from one timestep in a dataframe and tracks the evolution of the object across other timesteps.
    It uses a basic algorithm of finding a nearby halo of equivalent size from an adjacent timestep and considers it the same halo.

    Args:
        halo_id: Unique identifier for halo.
        timestep: The timestep that the halo can be found in.
        x_column_name: Name of the column containing the x coordinate.
        y_column_name: Name of the column containing the y coordinate.
        z_column_name: Name of the column containing the z coordinate.
        size_column_name: Name of the column containing the halo size.
        id_column_name: Name of the column containing the halo unique id.

    Return:
        str: Path to the dataframe that halo evolution was written to.
    """
    session_id = state.get("session_id", "")
    state_key = state.get("state_key", "")

    db_path = state.get("db_path", "")
    df_index = state.get("df_index", 0)
    results_list = state.get("results_list", [])

    distance_threshold = 2
    count_tolerance = 0.33

    conn = duckdb.connect(db_path)
    coord_columns = [x_column_name, y_column_name, z_column_name]

    timesteps = get_timesteps_from_db(conn)
    idx = timesteps.index(timestep)

    if timestep not in timesteps:
        raise ValueError(f"Timestep {timestep} not found in DB.")

    # Fetch target object
    row = get_object_from_db(conn, timestep, halo_id, id_column_name, coord_columns, size_column_name)
    if row is None:
        raise ValueError(f"Object {halo_id} not found at timestep {timestep}")
    current_center = np.array([row[0], row[1], row[2]])
    current_count = row[3]

    matched = [{
        "time_step": timestep,
        id_column_name: halo_id,
        x_column_name: current_center[0],
        y_column_name: current_center[1],
        z_column_name: current_center[2],
        size_column_name: current_count
    }]

    forward = track_direction(
        conn, timesteps[idx+1:], current_center, current_count,
        distance_threshold, count_tolerance, id_column_name, coord_columns, size_column_name
    )

    backward = track_direction(
        conn, reversed(timesteps[:idx]), current_center, current_count,
        distance_threshold, count_tolerance, id_column_name, coord_columns, size_column_name
    )

    all_entries = list(backward)[::-1] + matched + list(forward)
    compiled_df = pd.DataFrame(all_entries)
    pretty_output = pretty_print_df(compiled_df, max_rows=5)

    file = f"{WORKING_DIRECTORY}{session_id}/{session_id}_{df_index}.csv"
    compiled_df.to_csv(file, index=False)
    results_list.append((file, f"Calculated all halos equivalent to halo ID = {halo_id} for all timesteps using nearest neighbor in adjacent timesteps."))
    df_index +=1

    return Command(update={
        "results_list": results_list,
        "df_index": df_index,
        "stashed_msg": "SUCCESS",
        "messages": [
            ToolMessage(
                f"Called track_halo_evolution().\n\n{pretty_output}",
                tool_call_id=tool_call_id,
            )
        ]
    })


def track_direction(conn, timesteps, start_center, start_count, distance_threshold, count_tolerance, id_name, coord, size):
    results = []
    ref_center = np.array(start_center)
    ref_count = start_count

    for t in tqdm(timesteps, desc="Processing timesteps:"):
        try:
            df = get_data_from_db(conn, t, id_name, coord, size)
            if df.empty:
                continue

            coords = df[[coord[0], coord[1], coord[2]]].values
            counts = df[size].values
            tags = df[id_name].values

            dists = np.linalg.norm(coords - ref_center, axis=1)
            candidate_idxs = np.where(dists < distance_threshold)[0]

            matched = False
            for i in candidate_idxs:
                if abs(counts[i] - ref_count) / ref_count <= count_tolerance:
                    ref_center = coords[i]
                    ref_count = counts[i]
                    results.append({
                        "time_step": t,
                        id_name: tags[i],
                        coord[0]: ref_center[0],
                        coord[1]: ref_center[1],
                        coord[2]: ref_center[2],
                        size: ref_count
                    })
                    matched = True
                    break

            if not matched:
                break  # No good match, stop

        except Exception as e:
            print(f"[WARNING] Skipping timestep {t}: {e}")
            continue

    return results


def get_timesteps_from_db(conn):
    query = "SELECT DISTINCT time_step FROM haloproperties ORDER BY time_step ASC"
    return [row[0] for row in conn.execute(query).fetchall()]


def get_data_from_db(conn, timestep, id_name, coord, size):
    query = f"""
        SELECT {id_name}, {coord[0]}, {coord[1]}, {coord[2]}, {size}
        FROM haloproperties
        WHERE time_step = ?
    """
    return conn.execute(query, [timestep]).fetchdf()


def get_object_from_db(conn, timestep, object_id, id_name, coord, size):
    query = f"""
        SELECT {coord[0]}, {coord[1]}, {coord[2]}, {size}
        FROM haloproperties
        WHERE time_step = ? AND {id_name} = ?
    """
    return conn.execute(query, [timestep, object_id]).fetchone()

