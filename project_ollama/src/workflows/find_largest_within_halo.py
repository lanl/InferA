import sys
import pandas as pd
import numpy as np
from src.utils import json_loader
from src.utils.genericio_utils import read_gio_to_df
from src.utils.dataframe_utils import pretty_print_df
from src.workflows.base_workflow import BaseWorkflow


class FindLargestWithinHalo(BaseWorkflow):
    def __init__(self, base_path):
        """
        base_path: str - Base directory for analysis files, e.g. flamingo_B_1_analysis
        """
        self.base_path = base_path


    def run(self, object_type: str, timestep: int, halo_id:str, n: int, sort_by: str = None):
        """
        Entry point for running the workflow.

        Args:
            object_type: "halo" or "galaxy"
            timestep: simulation timestep
            n: number of largest objects to return
            sort_by: column to sort by, default depends on object_type
        """
        if object_type == "galaxy":
            sort_by = sort_by or "gal_count"
            df = self.get_n_objects_in_timestep_halo(object_type, timestep, halo_id, n, sort_by)

            print(f"✅ [find_largest_within_halo] task completed.")
            pretty_print_df(df)
            
            return df
        else:
            raise ValueError(f"Unsupported object_type: {object_type}")


    def get_n_objects_in_timestep_halo(self, object_type, timestep, halo:int, n, sort_by='gal_count'):
        # Sort galaxy properties values from highest to lowest and only get the top n
        df = read_gio_to_df(self.base_path, object_type, timestep, "properties")

        filtered_df = df[df['fof_halo_tag'] == halo]

        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        return filtered_df.head(n)