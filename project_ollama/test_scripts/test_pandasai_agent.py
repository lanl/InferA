import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from src.deprecated.preprocess_data import preprocess_data
from src.deprecated.param_extractor import extract_params
from src.llm.llm_client import llm, llm_raw
from src.deprecated import workflow_manager

import time
import threading
import uvicorn
from src.llm import fastapi_client
from utils.logger_config import setup_logger, get_logger

import pandas as pd
from pandasai import SmartDataframe

from src.utils import backend_duckdb

import pandas as pd
import ollama
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
from pandasai import Agent
from pandasai.connectors import PandasConnector
from typing import Optional

from src.utils.json_loader import get_variable_names_from_json, get_field_descriptions_from_json

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class OllamaLLM(LLM):
    """Custom LLM for using Ollama with PandasAI."""

    def __init__(self, model="mistral", api_url = None):
        self.model = model  # Default to Mistral
        self.api_url = api_url

    def call(self, instruction:str, context:Optional[str], **kwargs) -> str:
        """Send the prompt to Ollama and return the response."""
        prompt = str(instruction)
        if context:
            prompt += f"\nContext:\n{str(context)}"

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']

    @property
    def type(self) -> str:
        return "ollama_mistral"

def start_fastapi():
    uvicorn.run("llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

def main():
    # # Setup logger
    # ENABLE_LOGGING = False
    # ENABLE_DEBUG = False
    # setup_logger(ENABLE_LOGGING, ENABLE_DEBUG)
    # logging = get_logger(__name__)

    # # Start FastAPI server in background
    # server_thread = threading.Thread(target=start_fastapi, daemon=True)
    # server_thread.start()
    # logging.info("[INFO] FastAPI server running at http://127.0.0.1:8000")
    # print("[INFO] Sandbox started.")

    # params = {'task': 'find_largest_object', 'object_type': 'galaxy', 'timestep': 498, 'n': 10}

    # # Run workflow based on task classification and print result
    # result = workflow_manager.run_workflow(params["task"], params)
    # logging.info(f"[WORKFLOW OUTPUT] {result}")
    # print(f"[Workflow output] {result}")

    # print(type(result))
    # result.to_csv("test1.csv", index=False)

    dataset_key = "galaxyproperties"
    descriptions = get_field_descriptions_from_json(dataset_key)

    # description_str = "\n".join(
    #     f"{col}: {info['description'].lstrip('- ').strip()}"
    #     for col, info in variables.items()
    # )

    # csv_file_path ='example.csv'
    csv_file_path ='test1.csv'

    MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

    # binding function schema to LLM - temperature = 0 means LLM is deterministic
    ollama_llm= OllamaLLM(
        model=MODEL_NAME
        )

    df = pd.read_csv(csv_file_path)
    print(df)
    

    # Convert DataFrame into a SmartDataFrame
    # sdf = SmartDataframe(df, config={"llm": ollama_llm, : f"This contains the column names and the description of each column. Use descriptions to determine which column to query. Use column names to generate query code on those columns. Descriptions:\n{descriptions}", "save_logs": True, "verbose": True})
    # sdf = Agent(df, config={"llm": ollama_llm, "metadata": f"This contains the column names and the description of each column. Use descriptions to determine which column to query. Use column names to generate query code on those columns. Descriptions:\n{descriptions}", "save_logs": True, "verbose": True})
    connector = PandasConnector({"original_df": df}, field_descriptions = descriptions)
    sdf = SmartDataframe(connector, config={"llm": ollama_llm})
    # sdf = Agent(df, config={"llm": ollama_llm, "metadata": f"This contains information about things."})

    print("Simple Ollama AI Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            reply = sdf.chat(user_input)
            print("Ollama:", reply)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()