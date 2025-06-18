import os
import numpy as np
import pandas as pd

import requests
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.tools.load_data import load_data_tool

datapath_flamingo_B_1 = "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B/FSN_0.5387_VEL_149.279_TEXP_9.613_BETA_0.8710_SEED_9.548e5"
flamingo_B_1_analysis = datapath_flamingo_B_1 + "/analysis"
flamingo_B_1_output = datapath_flamingo_B_1 + "/output"


OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

tools = [load_data_tool]

# binding function schema to LLM - temperature = 0 means LLM is deterministic
llm= ChatOllama(
    model=MODEL_NAME,
    temperature = 0
)

llm_with_tools = llm.bind_tools(tools)

def ask_llm_with_tool():
    """ Simple ollama AI agent to test tool support """

    print("Simple Ollama AI Agent (type 'exit' to quit)")
    query = input("You: ")
    if query.lower() in {"exit", "quit"}:
        return

    # send user query
    messages = [HumanMessage(query)]
    response = llm_with_tools.invoke(messages)
    messages.append(response)
        
    df = pd.DataFrame()
    # llm responds with tool suggestions in response.tool_calls
    for tool_call in response.tool_calls:
        print(f" Tool requested by LLM: {tool_call["name"].lower()}")
        selected_tool = {"load_data_tool": load_data_tool}[tool_call["name"].lower()]

        print(f"Running tool with arguments: {tool_call['args']}")

        #asking LLM to generate function call with arguments
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        if tool_call["name"].lower() == "load_data_tool":
            df = tool_output

    #finally prompting LLM with custom function schema and all required arguments
    response = llm.invoke(messages)
    messages.append(response)

    print(df)

    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            break
        try:
            messages.append(query)
            response = llm.invoke(messages)
            print("Ollama: ", response)
            messages.append(response)

        except Exception as e:
            print("Error:", e)

# Simple loop to interact with the model
if __name__ == "__main__":
    ask_llm_with_tool()
