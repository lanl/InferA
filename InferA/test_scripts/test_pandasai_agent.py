import pandas as pd
import uvicorn

from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient
from pandasai import SmartDataframe

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.utils.config import (OPENAI_API_KEY, OPENAI_MODEL_NAME, LANLAI_API_TOKEN, LANLAI_MODEL_NAME, LANLAI_API_URL, PATH_TO_LANLCHAIN_PEM)

def start_fastapi():
    uvicorn.run("llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

def main():

    # llm = ChatOpenAI(
    #         api_key = LANLAI_API_TOKEN,
    #         model = LANLAI_MODEL_NAME,
    #         base_url= LANLAI_API_URL,
    #         http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM)
    #     )
    llm = ChatOpenAI(api_key = OPENAI_API_KEY, model_name = OPENAI_MODEL_NAME)

    csv_file_path ='src/data/df.csv'

    df = pd.read_csv(csv_file_path)
    print(df)

    sdf = SmartDataframe(df, config={"llm": llm})

    print("Simple Ollama AI Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            reply = sdf.chat(user_input)
            print("PandasAI:", reply)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()