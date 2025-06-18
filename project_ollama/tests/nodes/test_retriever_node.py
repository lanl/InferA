import pytest
import json

from langchain.prompts import PromptTemplate

from src.llm.llm_client import llm, embedding_model
from src.nodes.retriever_node import RetrieverNode

@pytest.fixture
def retriever_node():
    return RetrieverNode(embedding_model)

@pytest.fixture
def retriever_state(retriever_node):
    state = {}
    return retriever_node.run(state)

def test_build_vector_store(retriever_node):
    node = retriever_node
    state = {}
    node.run(state=state)
    vector_store = node.vector_store
    assert vector_store is not None

def test_run(retriever_state):
    assert "retriever" in retriever_state
    assert retriever_state["retriever"] is not None

def generate(retriever_state, question):
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an assistant for question-answer tasks about cosmology HACC data. Use the following pieces of retrieved context to retrieve one or more variable names that most matches the users query. Retrieved content is a description of the column names for various datasets. Answer with only the variable name, otherwise say you don't know.
                Question: {question}
                Context: {context}
                Answer:"""
            )
        retriever = retriever_state["retriever"]
        docs_content = retriever.invoke(question)
        messages = prompt.invoke({"question": question, "context": docs_content})
        response = llm.invoke(messages)
        return {"docs_content": docs_content, "prompt":prompt, "answer": response.content}

def test_generate(retriever_state):
    with open("src/data/RAG_QA.json") as f:
        data = json.load(f)

    qa_pairs = [(item['question'], item['expected_answer']) for item in data]

    for question, expected_answer in qa_pairs:
        llm_answer = generate(retriever_state, question)

        print(f"Question: {question}\nLLM answer: {llm_answer["answer"]}\nExpected answer: {expected_answer}")
            