# genericio_utils.py
# 
# This file contains useful functions from the genericio_explorer.ipynb notebook
# as well as various other utility functions for helping organize analysis_functions.py

import os
import sys
import numpy as np
import pandas as pd
from numba import njit, prange

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

from src.utils import json_loader

def list_genericio_files(folder_path: str, exclude_char: str = '#'):
    """
    Lists all files in a given folder that do not contain a specified character in their filenames.

    :param folder_path: Path to the folder
    :param exclude_char: Character to exclude files by (default is '#')
    :return: List of filenames that do not contain the exclude_char
    """
    if not os.path.isdir(folder_path):
        raise ValueError("The specified path is not a valid directory.")
    
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and exclude_char not in f]


def subselect_points(arr_list, sample_fraction = 0.1):
    """
    Randomly subsamples a given number of elements from each numpy array in a list

    :param arr_list: List of numpy arrays (all must have the same length)
    :param sample_faction: fraction to select
    :return: A new list of subsampled numpy arrays and number of points
    """
    if not arr_list:
        raise ValueError("The input list of arrays is empty.")

    array_lengths = [len(arr) for arr in arr_list]
    if len(set(array_lengths)) > 1:
        raise ValueError("All input arrays must have the same length.")
        
    total_points = len(arr_list[0])
    sample_size = int(sample_fraction * total_points)

    # Randomly sample indices
    indices = np.random.choice(total_points, sample_size, replace=False)

    # Randomly sample indices
    return [arr[indices] for arr in arr_list], sample_size    


@njit
def region_select(center, radius, buffer, x, y, z):
    '''Compare position for halos
    :center is array of x,y,z
    :radius is a distance
    :buffer is a distance
    :x, y, z are 3 numpy array of float coordinates
    '''

    match_list = [] # list of positions that could be candidates
    dist_limit = radius+buffer    
    num_items = len(x)

    for i in prange(num_items):
        temp_coord = np.array([x[i], y[i], z[i]])

        dist_x = np.abs(temp_coord[0] - center[0])
        dist_y = np.abs(temp_coord[1] - center[1])
        dist_z = np.abs(temp_coord[2] - center[2])
        dist = max(dist_x, dist_y, dist_z)

        if dist < (dist_limit):
            match_list.append(i)
            
    return match_list


def region_filter(filename, var_names, filter_pos, filter_value):
    '''
    Returns a list of points filtered by a value
    
    args:
        filename: name of the file to extact data from
        var_names: scalar variables to pull
        filer_pos: index into the array of the value we will use to compare
        filter_value: the value of the filter
    '''
    arr_list = gio.read(filename, var_names)
    indices = np.where( arr_list[filter_pos] == filter_value )

    return [arr[indices] for arr in arr_list]


def create_rankData_BB_List(filename):
    '''Create the list of bounding boxes for data ranks'''
    numDataRanks = gio.get_num_ranks(filename)
    dataDims = gio.get_dims(filename)
    dataOrigin = gio.get_physOrigin(filename)
    dataPhyDims = gio.get_physScale(filename)

    rankDimsX = float(dataPhyDims[0])/dataDims[0]
    rankDimsY = float(dataPhyDims[1])/dataDims[1]
    rankDimsZ = float(dataPhyDims[2])/dataDims[2]
    
    bb_list = []
    for r in range(numDataRanks):
        dataCoord = gio.get_coords(filename, r)
        x_min = rankDimsX*dataCoord[0] + dataOrigin[0]
        x_max = x_min + rankDimsX
        
        y_min = rankDimsY*dataCoord[1] + dataOrigin[1]
        y_max = y_min + rankDimsY
        
        z_min = rankDimsZ*dataCoord[2] + dataOrigin[2]
        z_max = z_min + rankDimsZ
        
        bb = np.array([x_min,y_min,z_min, x_max,y_max,z_max])
        bb_list.append(bb)
        
    return bb_list


def box_box_intersect(box1, box2):
    ''' Check if two boxes intersect, and returns True if they do'''
    
    ''' args: box = np.array(LL_x,LL_y,LL_z, UR_x, UR_y, UR_z)'''
    
    if box1[0] > box2[3] or box1[3] < box2[0]:
        return False
    elif box1[1] > box2[4] or box1[4] < box2[1]:
        return False
    elif box1[2] > box2[5] or box1[5] < box2[2]:
        return False
    else:
        return True


def check_fully_inside(box1, box2):
    ''' Check if box1 is fully inside box2'''
    ''' args: box = np.array(LL_x,LL_y,LL_z, UR_x, UR_y, UR_z)'''
    
    if box1[0]>=box2[0] and box1[3]<=box2[3]:
        if box1[1]>=box2[1] and box1[4]<=box2[4]:
            if box1[2]>=box2[2] and box1[5]<=box2[5]:
                return True
    return False


def create_pos_BB(pos, dims):
    '''Creates a bounding box given a position and a "radius"'''
    return np.array([pos[0]-dims[0], pos[1]-dims[1], pos[2]-dims[2],
                     pos[0]+dims[0], pos[1]+dims[1], pos[2]+dims[2]])


def compute_intersection(pos, bb_list, x_dim,y_dim,z_dim):
    '''Determine the indices that intersect the pos'''
    
    # Create a bounding box around the position of the halo
    bb_pos = create_pos_BB(pos, [x_dim, y_dim, z_dim])
    
    intersection_list = [] # list of all data ranks itersecting
    
    # Check intersection
    for i in range(len(bb_list)):
        if box_box_intersect(bb_pos, bb_list[i]):
            intersection_list.append(i)
            
            # # if fully inside, return only this one
            if check_fully_inside(bb_pos, bb_list[i]):
                break
            
    return intersection_list


def get_points_region(filename, vars, center, radius, buffer):
    '''Subselect a region of the data
    A faster version is available at: https://git.cels.anl.gov/hacc/genericio/-/blob/octree/legacy_python/tools/src/visParticleTrace.py
    '''
    
    # Get position of x, y, z
    index_x = -1
    index_y = -1
    index_z = -1
    
    for i, v in enumerate(vars):
        if v == 'x':
            index_x = i
            break
    if index_x == -1:
        for i, v in enumerate(vars):
            if '_x' in v:
                index_x = i
                break
    if index_x == -1:
        return
    
    for i, v in enumerate(vars):
        if v == 'y':
            index_y = i
            break
    if index_y == -1:
        for i, v in enumerate(vars):
            if '_y' in v:
                index_y = i
                break
    if index_y == -1:
        return
    
    for i, v in enumerate(vars):
        if v == 'z':
            index_z = i
            break
    if index_z == -1:
        for i, v in enumerate(vars):
            if '_z' in v:
                index_z = i
                break
    if index_z == -1:
        return
        
            
    bb_list = create_rankData_BB_List(filename)
    ranks_intersection_list = compute_intersection(center, bb_list, radius+buffer, radius+buffer, radius+buffer)
    
    particles_vals = gio.read_ranks(filename, vars, ranks_intersection_list)
    matched_indices = region_select(center, radius, buffer, particles_vals[index_x], particles_vals[index_y], particles_vals[index_z]) 
                                                
    return [arr[matched_indices] for arr in particles_vals]    


# def get_points_region(filename, vars, center, radius, buffer):
#     '''Subselect a region of the data.'''

#     def find_index(variants):
#         for axis in [axis for axis in ['x', 'y', 'z']]:
#             try:
#                 return variants.index(axis)
#             except ValueError:
#                 for i, v in enumerate(variants):
#                     if f"_{axis}" in v:
#                         return i
#             raise ValueError(f"Could not find {axis} coordinate in vars: {variants}")

#     try:
#         index_x = find_index(vars)
#         index_y = find_index(vars)
#         index_z = find_index(vars)
#     except ValueError as e:
#         print(f"[ERROR] {e}")
#         return None

#     try:
#         bb_list = create_rankData_BB_List(filename)
#         ranks_intersection_list = compute_intersection(center, bb_list, radius + buffer, radius + buffer, radius + buffer)

#         particles_vals = gio.read_ranks(filename, vars, ranks_intersection_list)
#         if not particles_vals or len(particles_vals) < 3:
#             print("[ERROR] Failed to load sufficient particle data")
#             return None

#         matched_indices = region_select(center, radius, buffer,
#                                         particles_vals[index_x],
#                                         particles_vals[index_y],
#                                         particles_vals[index_z])

#         if matched_indices is None:
#             print("[WARN] No matched points")
#             return None

#         return [arr[matched_indices] for arr in particles_vals]
    
#     except Exception as e:
#         print(f"[EXCEPTION] While processing file {filename}: {e}")
#         return None



def save_point_cloud_to_vtp(x, y, z, scalar_arrays=None, names=None, output_file="points.vtp"):
    """
    x, y, z: 1D numpy arrays of the same length (point coordinates)
    scalar_arrays: optional list of 1D arrays (same length) for point data
    names: names for scalar arrays
    """

    num_points = len(x)
    assert len(y) == num_points and len(z) == num_points

    # Create vtkPoints
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_points)
    for i in range(num_points):
        points.SetPoint(i, x[i], y[i], z[i])

    # Create vertices (VTK_VERTEX cells for each point)
    vertices = vtk.vtkCellArray()
    for i in range(num_points):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)

    # Create PolyData and assign points and cells
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)  # This is what makes points renderable


    # Add optional scalar arrays as point data
    if scalar_arrays and names:
        if len(scalar_arrays) != len(names):
            raise ValueError("Number of arrays and names must match.")
        for array, name in zip(scalar_arrays, names):
            vtk_array = numpy_support.numpy_to_vtk(array, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName(name)
            polydata.GetPointData().AddArray(vtk_array)
        polydata.GetPointData().SetActiveScalars(names[0])  # optional: set first as active

    # Write to .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()

    print(f"Saved point cloud to: {output_file}")


def read_output_file(output_path, timestep: int):
    file = output_path + f"/m000p.full.mpicosmo.{timestep}"
    output_vars = ['x', 'y', 'z', 'mass', 'rho', 'phi']
    values = gio.read(file, output_vars)
    return values


def read_gio_to_df(base_path, object_type: str, timestep: int, file_type: str, vars = None):
    """
    Read a file for the object_type into a dataframe.
    """
    file = base_path + f"/m000p-{timestep}.{object_type}{file_type}"

    if not vars:
        vars = json_loader.get_variable_names_from_json(f"{object_type}{file_type}")

    # Read file into a dataframe
    values = gio.read(file, vars)
    df = pd.DataFrame(np.column_stack(values), columns=vars)
    return df

def load_gio_to_df(file_path):
    """
    Read a file for the object_type into a dataframe.
    """
    ext = file_path.split(".")[-1]
    if not vars:
        vars = json_loader.get_variable_names_from_json(f"{ext}")

    # Read file into a dataframe
    values = gio.read(file_path, vars)
    df = pd.DataFrame(np.column_stack(values), columns=vars)
    return df