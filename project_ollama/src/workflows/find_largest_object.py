import sys
import pandas as pd
import numpy as np
from src.utils import json_loader
from src.utils.genericio_utils import read_gio_to_df
from src.utils.dataframe_utils import pretty_print_df
from src.workflows.base_workflow import BaseWorkflow


class FindLargestObject(BaseWorkflow):
    def __init__(self, base_path):
        """
        base_path: str - Base directory for analysis files, e.g. flamingo_B_1_analysis
        """
        self.base_path = base_path


    def run(self, object_type: str, timestep: int, n: int, sort_by: str = None):
        """
        Entry point for running the workflow.

        Args:
            object_type: "halo" or "galaxy"
            timestep: simulation timestep
            n: number of largest objects to return
            sort_by: column to sort by, default depends on object_type
        """
        if object_type == "halo":
            sort_by = sort_by or "fof_halo_count"
            df = self.get_n_largest_objects(object_type, timestep, n, sort_by)

        elif object_type == "galaxy":
            sort_by = sort_by or "gal_count"
            df = self.get_n_largest_objects(object_type, timestep, n, sort_by)
            
        else:
            raise ValueError(f"Unsupported object_type: {object_type}")
        
        print(f"✅ [find_largest_object] task completed.")
        pretty_print_df(df)
        return df


    def get_n_largest_objects(self, object_type: str, timestep: int, n: int, sort_by: str) -> pd.DataFrame:
        """
        Get top n largest objects sorted by sort_by at a given timestep.
        """
        df = read_gio_to_df(self.base_path, object_type, timestep, "properties")
        df = df.sort_values(by=sort_by, ascending=False)
        return df.head(n)