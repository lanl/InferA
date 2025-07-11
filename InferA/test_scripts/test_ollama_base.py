# test_ollama_base.py
#
# This file tests ollama ability to take queries regarding HACC data and interpret outputs from direct data input.
# Specifically tests on time step 498 and asks to find top 5 largest halos.
#
# Requires ollama server to be running

import sys
import requests
import os
import pandas as pd
import numpy as np
import time

from src.utils import genericio_utils

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)

import genericio as gio

datapath_flamingo_B_1 = "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B/FSN_0.5387_VEL_149.279_TEXP_9.613_BETA_0.8710_SEED_9.548e5"
flamingo_B_1_analysis = datapath_flamingo_B_1 + "/analysis"
flamingo_B_1_output = datapath_flamingo_B_1 + "/output"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

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

# Simple loop to interact with the model
if __name__ == "__main__":    

    haloproperties_498 = flamingo_B_1_analysis + "/m000p-498.haloproperties"

    halo_vars = ['fof_halo_count', 'sod_halo_radius', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']
    halo_values_498 = gio.read(haloproperties_498, halo_vars)
    halo_values_498_df = pd.DataFrame(np.column_stack(halo_values_498), columns=halo_vars)
    halo_values_498_df.sort_values(by='fof_halo_count', ascending=False, inplace=True) # we like seeing big halos
    halo_values_498_df.index.name = 'halo_index'
    top_20_halo_values_498_df = halo_values_498_df

    halo_values_text = top_20_halo_values_498_df.to_csv(index=True)

    print("Simple Ollama AI Agent (type 'exit' to quit)")
    try:
        user_query = "Read the csv and give me the 'halo_index' of the top 5 rows with the highest 'fof_halo_count' or 'sod_halo_radius. Also provide some commentary about what you noticed in the top 20 values.'\n"
        print("You: ", user_query)
        start = time.time()
        reply = ask_ollama(user_query + halo_values_text)
        print("Ollama:", reply)
        end = time.time()

        print(f"[TIMER] Execution took {end - start:.2f} seconds.")
    except Exception as e:
        print("Error:", e)
