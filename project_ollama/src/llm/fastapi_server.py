from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
import pandas as pd
import numpy as np
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Server is running"}

@app.post("/query/")
async def query_agent(request: Request, pandas_code: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"[SANDBOX SERVER] Received pandas_code: {pandas_code}")
    df = pd.read_csv(file.file)
    # Define the execution environment
    safe_globals = {
        '__builtins__': {},  # minimal built-ins (customize as needed)
        'pd': pd,
        'np': np
    }
    local_vars = {
        'input_df': df.copy()
    }

    try:
        exec(pandas_code, safe_globals, local_vars)
        result = local_vars.get("result_df")

        # result = eval(pandas_code, {"df": df, "pd": pd})
        logger.info(f"[SANDBOX SERVER] Pandas code executed successfully on dataframe. Result type: {type(result)}")
        
        # Convert the result to a JSON-serializable format
        if isinstance(result, pd.DataFrame):
            response = result.to_dict(orient='list')
        elif isinstance(result, pd.Series):
            response = result.to_dict(orient='list')
        else:
            response = str(result)
        return {"response": response}
    
    except Exception as e:
        logger.error(f"{str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        logger.info("[SANDBOX SERVER] Request processing completed")