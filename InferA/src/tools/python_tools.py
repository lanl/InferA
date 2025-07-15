from pydantic import BaseModel, Field

class GenerateCode(BaseModel):
    """Use this tool to generate code if other specific tools are not made to accomplish the task."""
    imports: str = Field(description = "Code Block import statements")
    python_code: str = Field(description = "Python code using pandas to process or analyze the input DataFrame, input_df.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")


class GenerateVisualization(BaseModel):
    """Use this tool to generate code if other specific tools are not made to accomplish the task."""
    imports: str = Field(description = "Code Block import statements")
    python_code: str = Field(description = "Python code to process the input DataFrame, input_df, into a pyvista.PolyData object written to a VTK-compatible file. Code block not including import statements.")
    explanation: str = Field(description = "Brief explanation of what the code is doing.")