import os
import pytest
from langchain_core.messages import AIMessage

from src.llm.llm_client import ChatLanlAI, ChatOllama  # Replace with actual import
from src.utils.config import ENABLE_LANLAI

# Replace with your actual config loading logic
LANLAI_ENABLED = os.getenv("ENABLE_LANLAI") == "true"

@pytest.mark.skipif(not ENABLE_LANLAI, reason="LanlAI not enabled or configured.")
def test_chat_lanlai_invoke():
    from openai import OpenAI
    from openai import DefaultHttpxClient  # Only if used
    from src.utils.config import LANLAI_API_TOKEN, LANLAI_API_URL, LANLAI_MODEL_NAME, PATH_TO_LANLCHAIN_PEM

    client = OpenAI(
        api_key=LANLAI_API_TOKEN,
        base_url=LANLAI_API_URL,
        http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM),
    )
    llm = ChatLanlAI(client=client, model_name=LANLAI_MODEL_NAME, temperature=0)

    response = llm.invoke("Hello")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0


@pytest.mark.skipif(ENABLE_LANLAI, reason="LanlAI enabled. Skipping ollama test.")
def test_chat_ollama_invoke():
    from src.utils.config import OLLAMA_MODEL_NAME  # Adjust accordingly
    from langchain_ollama import ChatOllama  # Assuming this is your base class

    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0, instruct=True)

    response = llm.invoke("Hello")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0
