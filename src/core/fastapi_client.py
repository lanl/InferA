"""
Module: dataframe_query_client.py
Purpose: This module enables sending pandas DataFrames along with transformation code to a
         remote FastAPI-based agent for execution. Useful in sandboxed environments or
         when offloading heavy or controlled pandas operations.

Functions:
    - query_dataframe_agent(dfs, pandas_code, imports, api_url): Sends DataFrames and code to the API agent and returns the result.
"""

import os
import requests
import pandas as pd
import tempfile
import logging
from typing import List


logger = logging.getLogger(__name__)

def query_dataframe_agent(dfs: List[pd.DataFrame], pandas_code: str, imports: str, api_url: str = "http://127.0.0.1:3000") -> str:
    """
    Send a DataFrame, pandas code, and imports to the FastAPI agent server and return the response.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrames to send.
        pandas_code (str): The pandas code to be executed on the server.
        imports (str): The import statements, each on a separate line.
        api_url (str): Base URL of the FastAPI server.

    Returns:
        str: The response from the agent.
    """
    try:
        files = {}
        temp_files = []

        # multiple dataframes can be loaded at once, each named input_df1, input_df2,... etc
        for i, df in enumerate(dfs):
            # Save each DataFrame to a temporary CSV file stored in files[]
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                temp_file_path = tmp.name
                df.to_csv(temp_file_path, index=False)
                temp_files.append(temp_file_path)

                file_param_name = f'file{i+1}'
                files[file_param_name] = open(temp_file_path, 'rb')

        # send code, imports and temp files to be processed
        data = {'pandas_code': pandas_code, 'imports': imports}
        logger.debug(f"[SANDBOX CLIENT] Preparing to send POST request. Data: {data}, df tempfile: {files}.")
        
        query_res = requests.post(f"{api_url}/query/", files=files, data=data)

        # After query is done, delete the temporary files
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"[SANDBOX CLIENT] Failed to delete temporary file: {e}")

        if query_res.status_code != 200:
            logger.error(f"Query failed. Status code: {query_res.status_code}")
            raise Exception(f"Query failed: {query_res.text}")

        # Read response as json
        response_json = query_res.json()

        # Check if there was an error in execution
        if "error_type" in response_json:
            error_type = response_json.get("error_type", "Unknown error")
            error_message = response_json.get("error_message", "No error message provided")
            problematic_code = response_json.get("problematic_code", "")
            line_number = response_json.get("line_number", "unknown")
            
            error_details = f"Error type: {error_type}\nLine {line_number}: {error_message}\nProblematic code: {problematic_code}"
            logger.error(f"[SANDBOX CLIENT] Code execution error: {error_details}")
            raise Exception(error_details)
        
        # Handle successful execution with different result types
        if "error" in response_json:
            logger.error(f"[SANDBOX CLIENT] Server reported an error: {response_json['error']}")
            raise Exception(response_json["error"])
        
        # Process the result based on its type
        result_type = response_json.get("type", "unknown")
        file_path = response_json.get("file_path", "")
        raw_result = response_json.get("raw_result", "")
        
        logger.debug(f"[SANDBOX CLIENT] Result type: {result_type}, file path: {file_path}")
        
        # Process different result types
        if result_type == "dataframe":
            result = pd.read_csv(file_path)
        elif result_type == "series":
            result = pd.read_csv(file_path).iloc[:, 0]  # Convert first column back to series
        elif result_type == "string":
            result = raw_result
        else:
            result = str(raw_result)
            
        
        # Clean up the temporary result file on the server
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"[SANDBOX CLIENT] Failed to delete result file: {e}. This is only a warning if intending to generate new csv file. Visualization agent does not produce result file.")
        
        # Return results
        logger.debug(f"[SANDBOX CLIENT] Query successful. Result type: {type(result)}")
        return result

    except Exception as e:
        logger.exception(f"[SANDBOX CLIENT] An error occurred in query_dataframe_agent: {str(e)}")
        raise

    finally:
        logger.info("[SANDBOX CLIENT] POST request to sandbox server completed")
