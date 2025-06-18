# load_file.py
# 
# This file is used as a langchain tool.
# It is used for loading MPI files.

import sys
import numpy as np
import pandas as pd

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)

import genericio as gio
# from . import genericio_utils as gio_utils
from langchain_core.tools import tool

datapath_flamingo_B_1 = "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B/FSN_0.5387_VEL_149.279_TEXP_9.613_BETA_0.8710_SEED_9.548e5"
flamingo_B_1_analysis = datapath_flamingo_B_1 + "/analysis"
flamingo_B_1_output = datapath_flamingo_B_1 + "/output"

@tool
def load_data_tool(timestep: str, object_type: str) -> list:
    """Use this to load all values from a file into a dataframe. Input should be based on the request: timestep as str(number) and object_type should be either 'halo' or 'galaxy'."""
    timestep = int(timestep)
    object_type = object_type.strip().lower()
    file = flamingo_B_1_analysis + f"/m000p-{timestep}.{object_type}properties"

    # Get all columns in file
    vars = gio.get_scalars(file)[1]

    values = gio.read(file, vars)
    df = pd.DataFrame(np.column_stack(values), columns=vars)
    df.sort_values(by="fof_halo_count", ascending=False, inplace=True)  # we like seeing big halos

    # return values
    return df
