import shutil
import pandas as pd
from typing import Any, Dict
from tabulate import tabulate

def pretty_print_df(df: pd.DataFrame, max_rows: int = 50, tablefmt: str = 'fancy_grid'):
    """
    Pretty prints a DataFrame using tabulate, trimmed to fit terminal size.

    Args:
        df (pd.DataFrame): The DataFrame to print.
        max_rows (int): Max number of rows to display (default: 20).
        tablefmt (str): Tabulate format (default: 'fancy_grid').
    """
    term_width = shutil.get_terminal_size((120, 20)).columns

    # Print summary
    print(f"DataFrame summary:")
    print(f"- Number of rows: {len(df)}")
    print(f"- Number of columns: {len(df.columns)}")
    print(f"- Columns: {list(df.columns)}\n")

    # # Limit rows
    df_trimmed = df.head(max_rows)

    # Estimate column widths
    def estimate_col_widths(columns):
        return sum(len(str(col)) + 5 for col in columns)  # +5 for padding/borders

    # Trim columns if too wide
    col_list = df.columns.tolist()
    while estimate_col_widths(col_list) > term_width and len(col_list) > 1:
        col_list.pop()  # remove last column

    df_final = df[col_list]

    # Tabulate and print
    print(tabulate(df_final, headers='keys', tablefmt=tablefmt, showindex=False))


def pretty_print_dict(data: Dict[Any, Any], max_items: int = 50, tablefmt: str = 'fancy_grid'):
    """
    Pretty prints a dictionary using tabulate, trimmed to fit terminal size.

    Args:
        data (dict): Dictionary to pretty-print.
        max_items (int): Maximum number of items to display (default: 50).
        tablefmt (str): Tabulate format (default: 'fancy_grid').
    """
    term_width = shutil.get_terminal_size((120, 20)).columns

    print("Dictionary summary:")
    print(f"- Number of items: {len(data)}\n")

    # Convert dictionary to a list of key-value pairs
    items = list(data.items())[:max_items]

    # Estimate column widths for key and value
    def estimate_row_widths(items):
        return max((len(str(k)) + len(str(v)) + 7) for k, v in items)  # +7 for borders/padding

    # If the content is too wide, truncate values
    adjusted_items = []
    for k, v in items:
        key_str = str(k)
        val_str = str(v)
        if estimate_row_widths([(k, v)]) > term_width:
            max_val_len = term_width - len(key_str) - 10
            if max_val_len > 0:
                val_str = str(v)[:max_val_len] + '...'
        adjusted_items.append((key_str, val_str))

    # Tabulate and print
    print(tabulate(adjusted_items, headers=["Key", "Value"], tablefmt=tablefmt, showindex=False))