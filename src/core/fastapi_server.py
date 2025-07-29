import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request

import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import scipy

import tempfile
import os
import uuid

import logging
import builtins
import sys

from src.utils.dataframe_utils import pretty_print_df

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Server is running"}

@app.post("/query/")
async def query_agent(request: Request, pandas_code: str = Form(...), imports: str = Form(...), file: UploadFile = File(...)):
    logger.debug(f"[SANDBOX SERVER] Received pandas_code.")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error(f"[SANDBOX SERVER] Error reading CSV file: {str(e)}")
        return {"error": f"File reading error: {str(e)}"}

    # Define the execution environment
    # Remove unsafe builtins
    safe_builtins = dict(builtins.__dict__)

    # Remove unsafe builtins
    for unsafe in ['open', 'eval', 'exec', 'compile', 'input', 'help', 'exit', 'quit']:
        safe_builtins.pop(unsafe, None)

    safe_globals = {
        '__builtins__': safe_builtins,  # minimal built-ins (customize as needed)
        '__import__': __import__, # Allow __import__
        # data analysis
        'pd': pd,
        'np': np,
        'scipy': scipy,
        # data visualization
        'plt': plt,
        'pv': pv,
        'vtk': vtk,
    }

    local_vars = {
        'input_df': df.copy()
    }

    # Main try-except block
    try:
        combined_code = f"{imports}\n\n{pandas_code}"

        exec(combined_code, safe_globals, local_vars)
        result = local_vars.get("result")

        logger.info(f"[SANDBOX SERVER] Pandas code executed successfully on server.")
        logger.debug(f"[SANDBOX SERVER]      - Result type: {type(result)}")

        # Create a temporary file to store the result
        temp_dir = tempfile.gettempdir()
        file_uuid = str(uuid.uuid4())
        file_path = os.path.join(temp_dir, f"{file_uuid}.csv")
        
        # Convert the result to a JSON-serializable format
        if isinstance(result, pd.DataFrame):
            logger.debug(f"[SANDBOX SERVER]      - Result DataFrame shape: {result.shape}\n")
            result.to_csv(file_path, index=False)
            return {"file_path": file_path, "type": "dataframe"}
        elif isinstance(result, pd.Series):
            pd.DataFrame(result).to_csv(file_path, index=False)
            return {"file_path": file_path, "type": "series"}
        elif isinstance(result, str):
            return {"raw_result": result, "file_path": "", "type": "string"}
        elif result is None:
            return {"raw_result": "", "file_path": "", "type": "none"}
        else:
            # For other types, convert to string
            return {"raw_result": result, "file_path": "", "type": "other"}

    except (NameError, AttributeError, ValueError, SyntaxError, Exception) as e:
        return handle_exception(e, combined_code)

    finally:
        logger.info("[SANDBOX SERVER] Request processing completed")


def extract_problematic_code(combined_code, line_number, context_lines=3):
    """Extract code context around the problematic line."""
    code_lines = combined_code.split('\n')
    start_line = max(0, line_number - context_lines)
    end_line = min(len(code_lines), line_number + context_lines)
    
    # Add line numbers to the code snippet
    return "\n".join([f"{i+start_line+1}: {line}" for i, line in 
                     enumerate(code_lines[start_line:end_line])])


def handle_exception(e, combined_code):
    """Common exception handling logic."""
    error_type = type(e).__name__
    error_message = str(e)
    
    if isinstance(e, SyntaxError):
        # Handle syntax errors specially since they have line/offset information
        line_number = e.lineno
        offset = e.offset
        text = e.text
        
        problematic_code = extract_problematic_code(combined_code, line_number)
        
        # Add a pointer to the exact error position
        if text and offset:
            code_lines = problematic_code.split('\n')
            # Find the line with the error (should be in the middle of the context)
            for i, line in enumerate(code_lines):
                if line.startswith(f"{line_number}:"):
                    # Calculate the position for the pointer
                    prefix_length = len(str(line_number)) + 2  # "line_number: "
                    pointer_line = " " * prefix_length + " " * (offset - 1) + "^"
                    code_lines.insert(i + 1, pointer_line)
                    break
            problematic_code = "\n".join(code_lines)
        
        error_message = f"There's a syntax error in the provided code: {error_message}"
    else:
        # For other exceptions, use traceback to find the line number
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        
        line_number = tb.tb_lineno
        problematic_code = extract_problematic_code(combined_code, line_number)
        
        # Add specific context based on error type
        if error_type == "NameError":
            error_message = f"{error_message}. This might be due to an undefined variable or function."
        elif error_type == "AttributeError":
            error_message = f"{error_message}. This might be due to calling a non-existent method or accessing a non-existent attribute."
        elif error_type == "ValueError":
            error_message = f"{error_message}. This might be due to invalid input or operation."
    
    logger.error(f"[SANDBOX SERVER] {error_type}: {error_message}", exc_info=True)
    
    # For general exceptions, include the full traceback
    if error_type not in ["NameError", "AttributeError", "ValueError", "SyntaxError"]:
        error_traceback = traceback.format_exc()
        error_message = error_traceback
    
    return {
        "error_type": error_type,
        "error_message": error_message,
        "problematic_code": problematic_code,
        "line_number": line_number
    }
