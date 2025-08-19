"""
Module: preprocess_node.py
Purpose: This module defines a preprocessing Node for handling raw HACC simulation output files. 
         It parses file metadata, extracts variable information, auto-generates descriptions 
         using a language model, and writes out structured JSON files used in downstream tasks.

Functions:
    - run(state): Main entrypoint to the node, managing setup, description generation, and JSON writing.
    - setup_json(base_dir, descriptions_exist): Parses simulation files and builds/updates a JSON schema of variables.
    - helper_describe_json_vars(): Extracts variables per file extension from the data_variables.json.
    - autogenerate_json_description(): Uses an LLM to create human-readable descriptions for each variable.
"""

import os
import sys
import re
import json
import glob
import logging

from src.nodes.node_base import NodeBase

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)
import genericio as gio

from config import DATA_DICT_PATH, SIMULATION_PATHS

logger = logging.getLogger(__name__)


# File patterns to match HACC output file extensions
file_patterns = ['m000p-*.haloparticles',
 'm000p-*.bighaloparticles',
 'm000p-*.coreparticles',
 'm000p-*.galaxyparticles',
 'm000p-*.galaxyparticles.subgrid',
 'm000p-*.galaxyproperties',
 'm000p-*.galaxypropertybins',
 'm000p-*.haloparticles',
 'm000p-*.haloparticletags',
 'm000p-*.haloproperties',
 'm000p-*.sodbighaloparticles',
 'm000p-*.sodbighaloparticles.subgrid']



class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("Preprocess")
        self.llm = llm
        self.base_dir = SIMULATION_PATHS[0] # Use one of the simulation paths as an example, since they all use the same file pattern at the leaf

    def run(self, state):
        """
        Main entrypoint for the preprocessing node.
        Creates necessary directories, sets up data_variables.json, and populates it with LLM-generated descriptions.
        """
        os.makedirs(DATA_DICT_PATH, exist_ok = True)
        os.makedirs(f"{DATA_DICT_PATH}descriptions/", exist_ok = True)

        # First-time setup: only if the schema file doesn't exist or is empty
        if not os.path.exists(f"{DATA_DICT_PATH}data_variables.json") or os.path.getsize(f"{DATA_DICT_PATH}data_variables.json")==0:
            logger.info("Writing initial data_variables.json...")
            self.setup_json(self.base_dir, False)
        else:
            logger.info("Initial data_variables.json already exists and is non-empty. Skipping initial setup.")
            return {
                "messages": [{
                    "role": "assistant", 
                    "content": "[PREPROCESS] Preprocessing already done - data_variables exists and non-empty.\nBeginning analysis."
                }], 
                "next": "Planner",
                "current": "Preprocess",
            }
        
        # If description JSONs haven't been generated yet, do so
        if not glob.glob(f"{DATA_DICT_PATH}descriptions/*.json"):
            logger.info("Generating descriptions JSON files...")
            self.autogenerate_json_description()
        else:
            logger.info("Descriptions JSON files already exist. Skipping autogenerate_json_description().")

        # Update the main schema with LLM-generated descriptions
        logger.info("Updating data_variables.json with descriptions...")
        self.setup_json(self.base_dir, True)

        with open(f"{DATA_DICT_PATH}data_variables.json", "r") as f:
            data_variables = json.load(f)

        return {
            "messages": [{
                "role": "assistant", 
                "content": "[PREPROCESS] Preprocessing done - auto-generated data descriptions to src/data/JSON/data_variables.json.\nBeginning analysis."
            }], 
            "next": "Planner",
            "current": "Preprocess",
        }
    
    def setup_json(self, base_dir: str, descriptions_exist: bool):
        """
        Scans simulation data files to extract variable names per extension and builds/updates a JSON schema file.

        Parameters:
        - base_dir (str): Path to simulation output files (e.g. flamingo_B_1_analysis)
        - descriptions_exist (bool): Indicates if variable description JSONs have already been generated.
        """
        if not os.path.isdir(base_dir):
            raise ValueError("The specified path is not a valid directory.")

        if not glob.glob(base_dir):
            raise ValueError("No files matching patterns found in base_dir")

        output_path = os.path.join(DATA_DICT_PATH,"data_variables.json")

        def pattern_to_regex(pattern):
            """
            Convert glob-style pattern like 'm000p-*.haloparticles' into regex.
            * represents a numeric wildcard.
            """
            regex = re.escape(pattern).replace(r'\*', r'\d+')
            return f'^{regex}$'

        def find_file_by_pattern(pattern, search_path):
            """
            Search recursively for first file matching a given pattern.
            """
            regex = re.compile(pattern_to_regex(pattern))
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if regex.fullmatch(file):
                        return os.path.join(root, file)
            return None

        # Group found files by file extension
        extension_map = {}
        for pattern in file_patterns:
            found_file = find_file_by_pattern(pattern, base_dir)
            if found_file:
                filename = os.path.basename(found_file)
                parts = filename.split(".")

                # Special handling for .subgrid extension
                extension = ".".join(parts[-2:]) if parts[-1] == "subgrid" else parts[-1]
                extension_map.setdefault(extension, []).append(found_file)

        with open("src/data/file_descriptions.json", "r") as f:
            file_descriptions = json.load(f)

        # Create json for variables from all extension patterns
        json_data = {}
        for extension, files in extension_map.items():
            sample_file = files[0]
            col_names = gio.get_scalars(sample_file)[1]
            
            json_data[extension] = {
                "description": file_descriptions[extension], 
                "columns":{
                    col: {
                        "type":"float",
                        "description":"",
                        "nullable": "False"
                    }
                    for col in col_names
                }
            }

            # If descriptions already exist, update the column descriptions
            if descriptions_exist:
                desc_path = f"{DATA_DICT_PATH}/descriptions/{extension}_descriptions.json"
                if os.path.exists(desc_path):
                    with open(desc_path, "r", encoding="utf-8") as f:
                        descriptions = json.load(f)
                    for var, desc in descriptions.items():
                        if var in json_data[extension]["columns"]:
                            json_data[extension]["columns"][var]["description"] = desc
        
        # Write the compiled schema
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent = 2)

        logger.info(f"[OK] Wrote schema for data variables to {output_path}")


    def helper_describe_json_vars(self):
        """
        Loads variable schema and returns structured lists of extensions, their descriptions, and variables.

        Returns:
        - all_extension: list of file extensions
        - all_description: list of file-level descriptions
        - all_vars: list of variable names per extension
        """
        with open(f"{DATA_DICT_PATH}data_variables.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_vars = []
        all_extension = []
        all_description = []

        for extension, data in data.items():
            all_extension.append(extension)
            all_description.append(data["description"])
            vars = list(data["columns"].keys())
            all_vars.append(vars)

        return all_extension, all_description, all_vars

    def autogenerate_json_description(self):
        """
        Uses LLM to generate descriptions for all variables found in each file type.
        Output is saved as one JSON file per extension in: {DATA_DICT_PATH}/descriptions/
        """
        all_extension, all_description, all_vars = self.helper_describe_json_vars()

        def chunk_list(lst, size):
            """Splits a list into chunks of given size."""
            for i in range(0, len(lst), size):
                yield lst[i:i + size]
        
        for extension, description, vars in zip(all_extension, all_description, all_vars):
            descriptions = {}
            chunk_size = 20 # Customize this based on LLM token limits
            chunks = list(chunk_list(vars, chunk_size)) if len(vars) > chunk_size else [vars]

            for i, chunk in enumerate(chunks):
                prompt = f"""
                    You are an expert cosmology data scientist using HACC data. 
                    This is a table describing '{extension}'. {description}, return a JSON object with clear descriptions for each variable based on the context of {extension}. Include {extension} as part of the description when relevant.
                    
                    Return only a valid JSON object, formatted as:
                    {{
                    "variable1": "description1",
                    "variable2": "description2",
                    ...
                    }}
                    Do not wrap it in triple quotes or code blocks.

                    Variables:
                    {chunk}
                """

                logger.info(f"Asking LLM to describe {extension} - chunk {i + 1}/{len(chunks)}")
                response = self.llm.invoke(prompt)

                try:
                    chunk_dict = json.loads(response.content)
                    if not isinstance(chunk_dict, dict):
                        raise ValueError("Expected a JSON object.")
                    descriptions.update(chunk_dict)
                except json.JSONDecodeError:
                    logger.error(f"⚠️ Failed to decode JSON for chunk {i + 1} of {extension}\nRaw response:\n", response.content)

            # Write the final combined result to a single .json file
            output_path = f"{DATA_DICT_PATH}descriptions/{extension}_descriptions.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, indent=2)
                print(f"✅ Descriptions for {extension} written to {output_path}")