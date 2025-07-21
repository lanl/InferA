import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GenerateCode(BaseModel):
    """Use this tool to generate code if other specific tools are not made to accomplish the task."""
    imports: str = Field(description = "Code Block import statements")
    python_code: str = Field(description = "Python code using pandas to process or analyze the input DataFrame, input_df.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")
    output_description: str = Field(description="Briefly describe what the output is and what it looks like.")


class GenerateVisualization(BaseModel):
    """Use this tool to generate code if other specific tools are not made to accomplish the task."""
    imports: str = Field(description = "Code Block import statements")
    python_code: str = Field(description = "Python code to process the input DataFrame, input_df, into a pyvista.PolyData object written to a VTK-compatible file. Code block not including import statements.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")
    output_description: str = Field(description="Briefly describe what the output is and what it looks like.")


class LoadDataframes(BaseModel):
    """Use this tool to load the appropriate data from the available csvs based on what is needed for the task and the data in the file."""
    data_paths: list = Field(description = "A list of paths to the csv files to be loaded.")

    

