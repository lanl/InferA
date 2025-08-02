import os
import sys
import logging
import pandas as pd
import numpy as np
import duckdb
from typing import Annotated
from tqdm import tqdm
import vtk
from vtk.util import numpy_support

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from src.utils.dataframe_utils import pretty_print_df

from config import WORKING_DIRECTORY


genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

logger = logging.getLogger(__name__)

@tool(parse_docstring=True)
def track_halo_evolution(halo_id: str, timestep: int, timestep_column_name: str, x_coord_column_name: str, y_coord_column_name: str, z_coord_column_name: str, size_column_name: str, id_column_name: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    This tool takes one target halo from one timestep in a dataframe and tracks the evolution of the object across other timesteps.
    It uses a basic algorithm of finding a nearby halo of equivalent size from an adjacent timestep and considers it the same halo.

    Args:
        halo_id: Unique identifier for target halo.
        timestep: The timestep that the target halo can be found in.
        timestep_column_name: Name of the column containing timesteps.
        x_coord_column_name: Name of the column containing the x coordinate.
        y_coord_column_name: Name of the column containing the y coordinate.
        z_coord_column_name: Name of the column containing the z coordinate.
        size_column_name: Name of the column containing the halo size.
        id_column_name: Name of the column containing the halo unique id.

    Return:
        str: Path to the dataframe that halo evolution was written to.
    """
    session_id = state.get("session_id", "")

    db_path = state.get("db_path", "")
    df_index = state.get("df_index", 0)
    results_list = state.get("results_list", [])

    distance_threshold = 2
    count_tolerance = 0.33

    conn = duckdb.connect(db_path)
    coord_columns = [x_coord_column_name, y_coord_column_name, z_coord_column_name]

    timesteps = get_timesteps_from_db(conn, timestep_column_name)
    idx = timesteps.index(timestep)

    if timestep not in timesteps:
        raise ValueError(f"Timestep {timestep} not found in DB.")

    # Fetch target object
    row = get_object_from_db(conn, timestep, timestep_column_name, halo_id, id_column_name, coord_columns, size_column_name)
    if row is None:
        raise ValueError(f"Object {halo_id} not found at timestep {timestep}")
    current_center = np.array([row[0], row[1], row[2]])
    current_count = row[3]

    matched = [{
        timestep_column_name: timestep,
        id_column_name: halo_id,
        x_coord_column_name: current_center[0],
        y_coord_column_name: current_center[1],
        z_coord_column_name: current_center[2],
        size_column_name: current_count
    }]

    forward = track_direction(
        conn, timesteps[idx+1:], current_center, current_count,
        distance_threshold, count_tolerance, timestep_column_name, id_column_name, coord_columns, size_column_name
    )

    backward = track_direction(
        conn, reversed(timesteps[:idx]), current_center, current_count,
        distance_threshold, count_tolerance, timestep_column_name, id_column_name, coord_columns, size_column_name
    )

    all_entries = list(backward)[::-1] + matched + list(forward)
    compiled_df = pd.DataFrame(all_entries)
    pretty_output = pretty_print_df(compiled_df, max_rows=5)

    file = f"{WORKING_DIRECTORY}{session_id}/{session_id}_{df_index}.csv"
    compiled_df.to_csv(file, index=False)
    results_list.append((file, f"Calculated all halos equivalent to halo ID = {halo_id} for all timesteps using nearest neighbor in adjacent timesteps.", "Track evolution tool: The output is a dataframe where each row is a timestep tracking a target halo."))
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


def track_direction(conn, timesteps, start_center, start_count, distance_threshold, count_tolerance, timestep_name, id_name, coord, size):
    results = []
    ref_center = np.array(start_center)
    ref_count = start_count

    for t in tqdm(timesteps, desc="Processing timesteps:"):
        try:
            df = get_data_from_db(conn, t, timestep_name, id_name, coord, size)
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
                        timestep_name: t,
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


def get_timesteps_from_db(conn, timestep_name):
    query = f"SELECT DISTINCT {timestep_name} FROM haloproperties ORDER BY {timestep_name} ASC"
    return [row[0] for row in conn.execute(query).fetchall()]


def get_data_from_db(conn, timestep, timestep_name, id_name, coord, size):
    query = f"""
        SELECT {id_name}, {coord[0]}, {coord[1]}, {coord[2]}, {size}
        FROM haloproperties
        WHERE {timestep_name} = ?
    """
    return conn.execute(query, [timestep]).fetchdf()


def get_object_from_db(conn, timestep, timestep_name, object_id, id_name, coord, size):
    query = f"""
        SELECT {coord[0]}, {coord[1]}, {coord[2]}, {size}
        FROM haloproperties
        WHERE {timestep_name} = ? AND {id_name} = ?
    """
    return conn.execute(query, [timestep, object_id]).fetchone()


@tool(parse_docstring=True)
def generate_pvd_file(timestep_column_name: str, x_coord_column_name: str, y_coord_column_name: str, z_coord_column_name: str, size_column_name: str, id_column_name: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    This tool takes a dataframe with multiple timesteps of an object and generates a .pvd collection file to visualize all the timesteps in ParaView. 
    Use this tool if user asks to visualize the evolution of an object over many timesteps.

    Args:
        timestep_column_name: Name of the column containing timesteps.
        x_coord_column_name: Name of the column containing the x coordinate.
        y_coord_column_name: Name of the column containing the y coordinate.
        z_coord_column_name: Name of the column containing the z coordinate.
        size_column_name: Name of the column containing the size metric.
        id_column_name: Name of the column containing the unique id.

    Return:
        str: Path to the dataframe that timestep visualization was written to.
    """
    session_id = state.get("session_id", "")

    df_index = state.get("df_index", 0)
    results_list = state.get("results_list", [])

    coord_columns = [x_coord_column_name, y_coord_column_name, z_coord_column_name]

    df_path = results_list[-1][0]
    df = pd.read_csv(df_path)
    pretty_print_df(df, max_rows=5)
    
    output_dir = f"{WORKING_DIRECTORY}{session_id}/{session_id}_{df_index}/"
    os.makedirs(output_dir, exist_ok = True)
    logger.debug(f"Created directory {output_dir}")

    grouped = df.groupby(timestep_column_name)

    vtp_files = []

    for timestep, group in grouped:
        points = vtk.vtkPoints()
        polydata = vtk.vtkPolyData()

        # Add points to the vtkPoints object
        for _, row in group.iterrows():
            points.InsertNextPoint(row[coord_columns[0]], row[coord_columns[1]], row[coord_columns[2]])
        
        polydata.SetPoints(points)

        # Create a vtkCellArray to store point data
        vertices = vtk.vtkCellArray()
        for i in range(len(group)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)

        # Add the vertices to the polydata
        polydata.SetVerts(vertices)

        # Add halo_count as a scalar attribute
        halo_counts = numpy_support.numpy_to_vtk(group[size_column_name].values)
        halo_counts.SetName(size_column_name)
        polydata.GetPointData().AddArray(halo_counts)

        # Add halo_tag as a scalar attribute
        halo_tags = numpy_support.numpy_to_vtk(group[id_column_name].values)
        halo_tags.SetName(id_column_name)
        polydata.GetPointData().AddArray(halo_tags)

        # Write the polydata to a .vtp file
        writer = vtk.vtkXMLPolyDataWriter()
        vtp_filename = f"timestep_{timestep}.vtp"
        vtp_path = os.path.join(output_dir, vtp_filename)
        writer.SetFileName(vtp_path)
        writer.SetInputData(polydata)
        writer.Write()

        vtp_files.append(vtp_filename)

    # Generate the PVD file
    pvd_path = os.path.join(output_dir, f"pvd_{df_index}.pvd")
    with open(pvd_path, "w") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for i, filename in enumerate(vtp_files):
            pvd.write(f'    <DataSet timestep="{i}" group="" part="0" file="{filename}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')

    logger.info(f"[GENERATE PVD FILE] PVD file generated: {pvd_path}")
    df_index += 1

    return Command(update={
        "df_index": df_index,
        "stashed_msg": "SUCCESS",
        "messages": [
            ToolMessage(
                f"Called generate_pvd_file().\nPVD file output to {pvd_path}",
                tool_call_id=tool_call_id,
            )
        ]
    })


