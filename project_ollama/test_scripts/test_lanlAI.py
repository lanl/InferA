import os
from dotenv import load_dotenv

from openai import DefaultHttpxClient, OpenAI
from langchain_openai import ChatOpenAI

from src.llm.llm_client import llm, token_tracker

load_dotenv()

# client = ChatOpenAI(
#     api_key = os.getenv("OPENAI_KEY"),
#     model = "gpt-4o-mini"
#     )

lanlAI_token = os.getenv("lanlAI_token")

base_url = "https://aiportal-api.stage.aws.lanl.gov/v2/serve/chat/completions"
MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"  # Or any model you've pulled


client = ChatOpenAI(
    api_key = lanlAI_token,
    model = MODEL_NAME,
    base_url="https://aiportal-api.stage.aws.lanl.gov/v2/serve",
    http_client=DefaultHttpxClient(verify="lanlchain.pem"),
)


# Simple loop to interact with the model
if __name__ == "__main__":
    completion = llm.invoke("Hello, how are you?")
    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "user", "content": "Tell a good joke in the form of a question. Do not yet give the answer."}
    #     ]
    # )
    completion = client.invoke("Hello, how are you?")
    # print(completion.choices[0].message.content)
    print(completion)
    print(type(completion))
    print(token_tracker.get_usage())
    
    # print(client.models.list())
    # completion = client.chat.completions.create(
    #     model= MODEL_NAME,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "How do I check if a Python object is an instance of a class?",
    #         },
    #     ],
    # )

    # print(completion)

        # completion = client.create(
    #     model= MODEL_NAME,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "How do I check if a Python object is an instance of a class?",
    #         },
    #     ],
    # )
    # print(completion.choices[0].message.content)
    # print(client.models.list())
    # print("Simple Ollama AI Agent (type 'exit' to quit)")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in {"exit", "quit"}:
    #         break
    #     try:
    #         reply = ask_lanlAI(user_input)
    #         print("Ollama:", reply)
    #     except Exception as e:
    #         print("Error:", e)
