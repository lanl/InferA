# write_QA_tests.py

# run this script using python -m scripts.write_QA_tests

import json
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm.llm_client import llm, embedding_model
from src.nodes.retriever_node import RetrieverNode

def generate_rag_text(fields):
    return "\n".join(
        f"Field: {field}\n  - Type: {props['type']}\n  - Description: {props['description']}\n  - Nullable: {props['nullable']}\n"
        for field, props in fields.items()
    )

def generate_questions_answers():
    """
    Generates 100 questions and answers in the format:
    {"question": question, "expected_answer": answer}
    """
    with open("src/data/JSON/data_variables.json") as f:
            schema = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_text = []
    for section, fields in schema.items():
        rag_text = generate_rag_text(fields)
        docs = text_splitter.split_text(rag_text)
        split_text.append(docs)
    

    questions_answers = []
    for _ in range(5):
        topic = random.choice(split_text)
        prompt = f"You are an expert data scientist on cosmology HACC simulation data. Pretend you are a scientist asking a single general question about simulation data where answering the question would utilize {topic} but do not include the variable names. Respond with only one question."
        response = llm.invoke(prompt)
        print(response.content)
        questions_answers.append({"question": response.content, "expected_answer": []})

    return questions_answers


def main():
    questions_answers = generate_questions_answers()

    # Save the result to a JSON file
    with open("src/data/RAG_QA.json", "w") as f:
        json.dump(questions_answers, f, indent=2)


if __name__ == "__main__":
    main()
