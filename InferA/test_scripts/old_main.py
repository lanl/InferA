# main.py
# 
# Main entry point for LLM software

from langchain_ollama import ChatOllama
# from src.data.preprocess_data import preprocess_data
# from deprecated.param_extractor import extract_params
from src.llm.llm_client import llm, llm_raw
from src.deprecated import workflow_manager
from src.llm import fastapi_client
from src.utils.logger_config import setup_logger, get_logger

import time
import threading
import uvicorn

from src.langgraph_class.states import AnalysisState
from src.langgraph_class.graph_builder import build_graph


datapath_flamingo_B_1 = "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B/FSN_0.5387_VEL_149.279_TEXP_9.613_BETA_0.8710_SEED_9.548e5"
flamingo_B_1_analysis = datapath_flamingo_B_1 + "/analysis"
flamingo_B_1_output = datapath_flamingo_B_1 + "/output"


def start_fastapi():
    uvicorn.run("src.llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


def main():
    # Setup logger
    ENABLE_LOGGING = True
    ENABLE_DEBUG = False
    setup_logger(ENABLE_LOGGING, ENABLE_DEBUG)
    logging = get_logger(__name__)


    # Start FastAPI server in background
    server_thread = threading.Thread(target=start_fastapi, daemon=True)
    server_thread.start()
    logging.info("[INFO] FastAPI server running at http://127.0.0.1:8000")
    print("[INFO] Sandbox started.")
    time.sleep(1) 

   # while True:
    #     try:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["quit", "exit", "q"]:
    #             print("Goodbye!")
    #             break
    #         stream_graph_updates(user_input)
    #     except:
    #         # fallback if input() is not available
    #         user_input = "What do you know about LangGraph?"
    #         print("User: " + user_input)
    #         stream_graph_updates(user_input)
    #         break
    # while True:
    #     # User input
    #     input_prompt = "\nYou - Enter your data query:      "
    #     user_query = input(input_prompt)
    #     if user_query.lower() in {"exit", "quit"}:
    #         break
    #     logging.info(f"[USER QUERY] {user_query}")
    #     # user_query = "Find the top 5 largest halo in timestep 498."

    #     message = [user_query]
        
    #     # Set params and missing to be empty so that after work is completed, start work from fresh.
    #     params, missing = extract_params(llm, user_query)

    #     while not params or missing:
    #         # Request additional information - it could be a different task or adding additional information required for the workflow.
    #         original_task = params["task"]
    #         input_prompt = (f"\nIt looks like you would like to use the workflow [{original_task}]. However required information is missing for the workflow.\nCan you provide the following? {', '.join(missing)}:     ")
    #         logging.info(f"[ROUTING AGENT]\nTask workflow: {original_task}\nParams required: {params}\nParams missing: {missing}")
    #         user_query = input(input_prompt)
    #         logging.info(f"[USER QUERY] {user_query}")
            
    #         # In first loop, this adds the initial query and additional query
    #         message.append(f"IMPORTANT TASK: {user_query}")
            
    #         params, missing = extract_params(llm, message)
    #         new_task = params["task"]
    #         logging.info(f"[ROUTING AGENT]\nTask workflow: {original_task}\nParams required: {params}\nParams missing: {missing}")

    #         # If the task from the newest query is the same as original task, then keep appending messages so LLM gets more information.
    #         # Else if the task from newest query is different than original task, rewrite message with just task from newest query.
    #         if original_task == new_task:
    #             message.append(user_query)
    #         else:
    #             message = [user_query]

    #     print(params)
    #     print(params["task"])

    #     # Run workflow based on task classification and print result
    #     result = workflow_manager.run_workflow(params["task"], params)
    #     logging.info(f"[WORKFLOW OUTPUT] {result}")
    #     print(f"[Workflow output] {result}")

    #     # Ask user if they want to explore data. This enables dataframe explorer agent.
    #     user_explore = input('Do you want to continue exploring the data? Code will be run from a sandboxed FastAPI server (http://127.0.0.1:8000): (y/n)').lower().strip() == 'y'
        
    #     if user_explore:
    #         logging.info(f"[INFO] Dataframe explorer agent started")
    #         print("Starting DataFrame Explorer Agent (type 'exit' to quit)")
    #         while user_explore:
    #             user_query = input("What would you like to explore in the data?")
    #             logging.info(f"[USER QUERY] {user_query}")
    #             if user_query.lower() in {"exit", "quit"}:
    #                 break
    #             try:
    #                 response = fastapi_client.query_dataframe_agent(result, user_query)
    #                 logging.info(f"[DATAFRAME EXPLORER] {response}")
    #                 print(response)
    #             except Exception as e:
    #                 print("Error:", e)