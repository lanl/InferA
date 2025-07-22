import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request

import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import scipy

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

    try:
        combined_code = f"{imports}\n\n{pandas_code}"
    
        exec(combined_code, safe_globals, local_vars)
        result = local_vars.get("result")

        # result = eval(pandas_code, {"df": df, "pd": pd})
        logger.info(f"[SANDBOX SERVER] Pandas code executed successfully on server.")
        logger.debug(f"[SANDBOX SERVER]      - Result type: {type(result)}")
        
        # Convert the result to a JSON-serializable format
        if isinstance(result, pd.DataFrame):
            logger.debug(f"[SANDBOX SERVER]      - Result DataFrame shape: {result.shape}\n")
            response = result.to_dict(orient='records')
            # response = result.to_dict(orient='list')
        elif isinstance(result, pd.Series):
            response = result.to_dict()
        elif isinstance(result, str):
            response = result
        elif result is None:
            response = None
        else:
            response = str(result)
        return {"response": response}
    
    except NameError as e:
        logger.error(f"[SANDBOX SERVER] NameError: {str(e)}", exc_info=True)
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": f"NameError: {str(e)}. This might be due to an undefined variable or function."
                }
            }
    
    except AttributeError as e:
        logger.error(f"[SANDBOX SERVER] AttributeError: {str(e)}", exc_info=True)
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": f"AttributeError: {str(e)}. This might be due to calling a non-existent method or accessing a non-existent attribute."
                }
            }
    
    except ValueError as e:
        logger.error(f"[SANDBOX SERVER] ValueError: {str(e)}", exc_info=True)
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": f"ValueError: {str(e)}. This might be due to invalid input or operation."
                }
            }
    
    except SyntaxError as e:
        logger.error(f"[SANDBOX SERVER] SyntaxError: {str(e)}", exc_info=True)
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": f"There's a syntax error in the provided code: {str(e)}"
                }
            }
    except Exception as e:
        logger.error(f"[SANDBOX SERVER] Execution error: {str(e)}", exc_info=True)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": error_traceback
                }
            }
    except Exception as e:
        logger.error(f"[SANDBOX SERVER] Execution error: {str(e)}", exc_info=True)
        return {
            "response": {
                "error_type": type(e).__name__, 
                "error_message": f"Code execution error: {str(e)}"
                }
            }
    
    finally:
        logger.info("[SANDBOX SERVER] Request processing completed")