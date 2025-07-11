import os
import pandas as pd

import time
import threading
import uvicorn
from typing import Optional

import ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


from src.deprecated.preprocess_data import preprocess_data
from src.deprecated.param_extractor import extract_params
from src.deprecated import workflow_manager
from src.llm.llm_client import llm, llm_raw
from src.llm import fastapi_client
from src.utils.logger_config import setup_logger, get_logger
from src.utils import backend_duckdb
from src.utils.json_loader import get_variable_names_from_json, get_field_descriptions_from_json, extract_code_block

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def run_pandas_function_call(question: str, df):
    prompt = PromptTemplate(
        template = """
        You are a helpful assistant that receives a question about a Pandas dataframe named `df`.
        Generate a single line of Python code that operates on `df` to answer the question.

        Question: {question}

        Respond with only the pandas code (e.g. df.describe(), df[df['age'] > 30])
        """, 
        input_variables=["question"]
    )

    # Ask Ollama to generate pandas code
    response = (prompt | llm).invoke({"question": question})
    pandas_code = extract_code_block(response.content)
    print("Generated code:", pandas_code)

    # Safety check (very important to avoid executing arbitrary code)
    if not pandas_code.startswith("df"):
        raise ValueError("Generated code does not start with 'df', refusing to execute for safety.")

    # Evaluate the code on df
    # Use eval but limit globals and locals for safety in a real system you would sandbox more
    try:
        result = eval(pandas_code, {"df": df, "pd": pd})
    except Exception as e:
        return f"Error executing code: {e}"

    return result


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

    print(extract_code_block("```hello```"))

    print("Simple Ollama AI Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            print("Getting function definition...")
            reply = run_pandas_function_call(user_input, df)
            print("Ollama:", reply)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()