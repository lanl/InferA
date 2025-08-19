"""
Module: dataframe_utils.py
Purpose: Provides utility functions for formatting and printing `pandas` DataFrames and Python dictionaries 
         in a clean, readable format using the `tabulate` library.

Functions:
    - pretty_print_df(df, max_rows, tablefmt, return_output): Prints or returns a truncated and formatted view of a DataFrame.
    - pretty_print_dict(data, max_items, tablefmt, return_output): Prints or returns a formatted view of a dictionary with key-value pairs.
"""


import shutil
import pandas as pd
from typing import Any, Dict
from tabulate import tabulate

pd.set_option('display.float_format', '{:.15f}'.format)

def pretty_print_df(
    df: pd.DataFrame, 
    max_rows: int = 50, 
    tablefmt: str = 'fancy_grid',
    return_output: bool = False
) -> str | None:
    """
    Pretty prints a DataFrame using tabulate, trimmed to fit terminal size.
    
    Args:
        df (pd.DataFrame): The DataFrame to print.
        max_rows (int): Max number of rows to display (default: 50).
        tablefmt (str): Tabulate format (default: 'fancy_grid').
        return_output (bool): If True, returns output string instead of printing.
        
    Returns:
        str | None: Formatted output string if return_output is True, else None.
    """
    term_width = shutil.get_terminal_size((120, 20)).columns
    output_lines = []

    # Build summary
    output_lines.append("DataFrame summary:")
    output_lines.append(f"- Number of rows: {len(df)}")
    output_lines.append(f"- Number of columns: {len(df.columns)}")
    output_lines.append(f"- Columns: {list(df.columns)}\n")

    # Limit rows
    df_trimmed = df.head(max_rows)

    # Estimate column widths
    def estimate_col_widths(columns):
        return sum(len(str(col)) + 5 for col in columns)  # +5 for padding/borders

    # Trim columns if too wide
    col_list = df.columns.tolist()
    while estimate_col_widths(col_list) > term_width and len(col_list) > 1:
        col_list.pop()

    df_final = df_trimmed[col_list]

    # Create tabulated string
    table_str = tabulate(df_final, headers='keys', tablefmt=tablefmt, showindex=False)
    output_lines.append(table_str)

    # Add indication if there are more rows than displayed
    if len(df) > max_rows:
        output_lines.append(f"... (showing {max_rows} of {len(df)} rows)\n")

    final_output = "\n".join(output_lines)

    if return_output:
        return final_output
    else:
        print(final_output)


def pretty_print_dict(
    data: Dict[Any, Any], 
    max_items: int = 50, 
    tablefmt: str = 'fancy_grid', 
    return_output: bool = False
) -> str | None:
    """
    Pretty prints a dictionary using tabulate, trimmed to fit terminal size.

    Args:
        data (dict): Dictionary to pretty-print.
        max_items (int): Maximum number of items to display (default: 50).
        tablefmt (str): Tabulate format (default: 'fancy_grid').
        return_output (bool): If True, returns output string instead of printing.

    Returns:
        str | None: Formatted string if return_output is True, else None.
    """
    term_width = shutil.get_terminal_size((120, 20)).columns
    output_lines = []

    output_lines.append("Dictionary summary:")
    output_lines.append(f"- Number of items: {len(data)}\n")

    # Get subset of items
    items = list(data.items())[:max_items]

    # Estimate row width
    def estimate_row_widths(items):
        return max((len(str(k)) + len(str(v)) + 7) for k, v in items) if items else 0

    # Adjust items to fit terminal width
    adjusted_items = []
    for k, v in items:
        key_str = str(k)
        val_str = str(v)
        if estimate_row_widths([(k, v)]) > term_width:
            max_val_len = term_width - len(key_str) - 10
            if max_val_len > 0:
                val_str = val_str[:max_val_len] + '...'
        adjusted_items.append((key_str, val_str))

    table_str = tabulate(adjusted_items, headers=["Key", "Value"], tablefmt=tablefmt, showindex=False)
    output_lines.append(table_str)

    # Add indication if there are more rows than displayed
    if len(data) > max_items:
        output_lines.append(f"\n... (showing {max_items} of {len(data)} rows)")

    final_output = "\n".join(output_lines)

    if return_output:
        return final_output
    else:
        print(final_output)