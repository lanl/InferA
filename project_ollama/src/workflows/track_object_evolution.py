import os
import re
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import json_loader
from src.utils.genericio_utils import read_gio_to_df, list_genericio_files
from src.utils.dataframe_utils import pretty_print_df
from src.workflows.base_workflow import BaseWorkflow

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)

import genericio as gio

center_vars = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']
meta_vars = ["fof_halo_count", "sod_halo_radius", "fof_halo_tag"]
all_vars = center_vars + meta_vars

class TrackObjectEvolution(BaseWorkflow):
    def __init__(self, base_path):
        """
        Args:
            base_path (str): Directory containing the files named like /m000p-{timestep}.{object_type}{file_type}
        """
        self.base_path = base_path

    def run(self, object_type: str, object_id: str, timestep: str, distance_threshold = 1.5, count_tolerance=0.44):
        file_type = "properties"

        if object_type == "halo":
            tag = "fof_halo_tag"
        elif object_type == "galaxy":
            tag = "galaxy_tag"
            raise ValueError("Cannot accept galaxy as input for tracking evolution.")

        
        timesteps = sorted([
            int(re.search(r'm000p-(\d+)', f).group(1))
            for f in list_genericio_files(self.base_path)
            if f.endswith(f".{object_type}{file_type}")
        ])

        if timestep not in timesteps:
            raise ValueError(f"Timestep {timestep} not found.")

        idx = timesteps.index(timestep)

        # Step 2: Read target halo info at initial timestep
        df = read_gio_to_df(self.base_path, object_type, timestep, file_type, vars=all_vars)
        row = df[df[tag] == object_id]
        if row.empty:
            raise ValueError(f"Halo ID {object_id} not found in timestep {timestep}")

        current_center = row[center_vars].values[0]
        current_count = row["fof_halo_count"].values[0]

        matched = [{
            "timestep": timestep,
            tag: object_id,
            "fof_halo_center_x": current_center[0],
            "fof_halo_center_y": current_center[1],
            "fof_halo_center_z": current_center[2],
            "fof_halo_count": current_count
        }]

        # Track forward and backward
        forward = self.track_direction(
            timesteps[idx+1:], object_type, file_type, current_center, current_count,
            distance_threshold, count_tolerance
        )
        backward = self.track_direction(
            reversed(timesteps[:idx]), object_type, file_type, current_center, current_count,
            distance_threshold, count_tolerance
        )
        all_entries = backward[::-1] + matched + forward

        compiled_df = pd.DataFrame(all_entries)
        pretty_print_df(compiled_df)
        list_of_dfs = [pd.DataFrame([entry]) for entry in all_entries]

        print(f"✅ [track_object_evolution] task completed. This returns a list of dataframes, each one is a timestep of the target halo (candidate based on approximate tracing algorithm).")

        return list_of_dfs, compiled_df


    def track_direction(self, timesteps, object_type, file_type,
                        start_center, start_count,
                        distance_threshold, count_tolerance):
        """
        Tracks halo in one direction (forward or backward).

        Returns:
            List of matched halo records (dicts).
        """
        results = []
        ref_center = start_center
        ref_count = start_count

        for t in tqdm(timesteps, desc = "Processing timesteps:"):
            try:
                df = read_gio_to_df(self.base_path, object_type, t, file_type, vars=all_vars)
                coords = df[center_vars].values
                counts = df["fof_halo_count"].values
                tags = df["fof_halo_tag"].values

                dists = np.linalg.norm(coords - ref_center, axis=1)
                within = dists < distance_threshold

                candidate_idxs = np.where(within)[0]
                for i in candidate_idxs:
                    if abs(counts[i] - ref_count) / ref_count <= count_tolerance:
                        ref_center = coords[i]
                        ref_count = counts[i]
                        results.append({
                            "timestep": t,
                            "fof_halo_tag": tags[i],
                            "fof_halo_center_x": coords[i][0],
                            "fof_halo_center_y": coords[i][1],
                            "fof_halo_center_z": coords[i][2],
                            "fof_halo_count": counts[i]
                        })
                        break  # Stop after first match
                else:
                    break  # No good match found, stop

            except Exception as e:
                print(f"[WARNING] Skipping timestep {t}: {e}")
                continue
                
        return results
