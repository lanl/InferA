from langchain_core.tools import tool
import pandas as pd

@tool(parse_docstring=True)
def load_data(timestep: int, object: str) -> pd.DataFrame:
    """Load data from a simulation file based on simulation name, timestep, and object to load. 'timestep' and 'object' must be explicitly stated. 
    'object' can be one of the following: [haloproperties, galaxyproperties, haloparticles, galaxyparticles]
    
    Args:
        timestep: Timestep of simulation to load from.
        object: object type in simulation to load data about.
    """
    df = pd.DataFrame()
    return df

@tool(parse_docstring=True)
def filter_data(data_key: str, condition: str, state: dict) -> pd.DataFrame:
    """Filter a DataFrame using a condition string like 'id == 123'.
    
    Args:
        data_key: data_key in state to retrieve dataframe to use
        condition: Condition to filter dataframe using query.
    """
    data = state.get(data_key)
    return data.query(condition)

@tool(parse_docstring=True)
def summarize_data(data_key: str, state: dict) -> dict:
    """Summarize the DataFrame with basic statistics.
    
    Args:
        data_key: data_key in state to retrieve dataframe to use
    """
    data = state.get(data_key)
    return data.describe().to_dict()

@tool(parse_docstring=True)
def plot_data(data_key: str, x: str, y: str, z: str) -> dict:
    """Plot data from columns defined by file type to plot 3D points.
    
    Args:
        data_key: data_key in state to retrieve dataframe to use
        x: Column name for x coordinate.
        y: Column name for y coordinate.
        z: Column name for z coordinate.
    """
    return {"plot": f"chart of {y} over {x}"}


TOOL_REGISTRY = [load_data, filter_data, summarize_data, plot_data]