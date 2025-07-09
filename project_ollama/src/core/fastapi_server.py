from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import logging
import builtins

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Server is running"}

@app.post("/query/")
async def query_agent(request: Request, pandas_code: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"[SANDBOX SERVER] Received pandas_code.")
    df = pd.read_csv(file.file)
    # Define the execution environment
    # Remove unsafe builtins
    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith('_')}

    # Remove unsafe builtins
    for unsafe in ['open', 'eval', 'exec', 'compile', 'input', 'help', 'exit', 'quit', '__import__']:
        safe_builtins.pop(unsafe, None)

    safe_globals = {
        '__builtins__': safe_builtins,  # minimal built-ins (customize as needed)
        'pd': pd,
        'np': np,
        'plt': plt,
        'pv': pv
    }
    local_vars = {
        'input_df': df.copy()
    }

    try:
        exec(pandas_code, safe_globals, local_vars)
        result = local_vars.get("result")

        # result = eval(pandas_code, {"df": df, "pd": pd})
        logger.info(f"[SANDBOX SERVER] Pandas code executed successfully on dataframe.")
        logger.info(f"[SANDBOX SERVER]      - Result type: {type(result)}")
        
        # Convert the result to a JSON-serializable format
        if isinstance(result, pd.DataFrame):
            logger.info(f"[SANDBOX SERVER]      - Result DataFrame shape: {result.shape}\n\nResult:\n{result}")
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