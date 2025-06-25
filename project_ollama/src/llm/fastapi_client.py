import json
import requests
import pandas as pd
import tempfile
import logging

logger = logging.getLogger(__name__)

def query_dataframe_agent(df: pd.DataFrame, pandas_code: str, api_url: str = "http://127.0.0.1:8000") -> str:
    """
    Send a DataFrame and a prompt to the FastAPI agent server and return the response.

    Args:
        df (pd.DataFrame): The DataFrame to send.
        pandas_code (str): The pandas code to be executed on the server.
        api_url (str): Base URL of the FastAPI server.

    Returns:
        str: The response from the agent.
    """
    try:
        # Save the DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            with open(tmp.name, 'rb') as file:
                files = {'file': file}
                data = {'pandas_code': pandas_code}
                logger.info(f"[SANDBOX CLIENT] Preparing to send POST request. Data: {data}, df tempfile: {files}.")
                query_res = requests.post(f"{api_url}/query/", files=files, data=data)

        if query_res.status_code != 200:
            logger.error(f"Query failed. Status code: {query_res.status_code}")
            raise Exception(f"Query failed: {query_res.text}")

        response = query_res.json().get("response", "")
        logger.info(f"[SANDBOX CLIENT] Query successful. Response length: {len(str(response))}")
        return response

    except Exception as e:
        logger.exception(f"An error occurred in query_dataframe_agent: {str(e)}")
        raise

    finally:
        logger.info("[SANDBOX CLIENT] query_dataframe_agent function completed")
