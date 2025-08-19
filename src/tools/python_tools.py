"""
Module: schemas.py
Purpose: This module defines Pydantic models used for structured code generation, data visualization,
         and loading data files in an LLM-assisted data analysis pipeline.

Models:
    - GenerateCode: Structured input/output format for generating pandas-based data processing code.
    - GenerateVisualization: Structured input/output format for generating 3D data visualizations.
    - LoadDataframes: Used to specify CSV file paths to be loaded into dataframes.
"""

import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerateCode(BaseModel):
    """
    Use this model when generating pandas-based data transformation or analysis code.
    Typically used when no other specific code generation tool fits the task.
    """
    imports: str = Field(description = "Code Block import statements.")
    load_csv_code: str = Field(description = "Code to load csv into a pandas dataframe called input_df.")
    python_code: str = Field(description = "Python code using pandas to process or analyze the input DataFrame, input_df.")
    result_output_code: str = Field(description = "Code to produce an output dataframe called 'result'. This runs after all other python code has run. It should look like 'result = ...'.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")
    output_description: str = Field(description="Briefly describe what the output is and what it looks like.")


class GenerateVisualization(BaseModel):
    """
    Use this model to structure code that generates 3D visualizations from tabular data.
    Intended for tools that use libraries like PyVista for mesh generation or VTK-compatible outputs.
    """
    imports: str = Field(description = "Code Block import statements")
    python_code: str = Field(description = "Python code to process the input DataFrame, input_df, into a pyvista.PolyData object written to a VTK-compatible file. Code block not including import statements.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")
    output_description: str = Field(description="Briefly describe what the output is and what it looks like.")


class LoadDataframes(BaseModel):
    """
    Use this model to specify a list of CSV file paths to be loaded into pandas DataFrames.
    The loading behavior is typically automated based on the task's requirements.
    """
    data_paths: list = Field(description = "A list of paths to the csv files to be loaded.")

    

