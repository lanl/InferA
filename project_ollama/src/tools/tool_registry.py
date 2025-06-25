from langchain_core.tools import tool
import pandas as pd

@tool
def filter_dataframe(df: pd.DataFrame, column: str, condition: str) -> pd.DataFrame | str:
    """
    Filter a DataFrame based on a condition applied to a column.

    Args:
        df: The input DataFrame to filter.
        column: Column name to apply the condition on.
        condition: A string condition, like '> 10', '== "star"', '<= 5'.

    Returns:
        Filtered DataFrame, or an error message string if invalid.
    """
    if column not in df.columns:
        return f"ColumnNotFound: {column}"

    try:
        # Build and evaluate the query string
        query_str = f"`{column}` {condition}"
        filtered_df = df.query(query_str)
        return filtered_df
    except Exception as e:
        return f"FilterError: {e}"


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


# TOOL_REGISTRY = [load_data, filter_data, summarize_data, plot_data]