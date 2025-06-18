import json

from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import START, StateGraph

from typing_extensions import List, TypedDict

from src.llm.llm_client import llm


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = InMemoryVectorStore(embeddings)

    # Load the JSON schema from file
    with open('src/data/JSON/data_variables.json') as f:
        schema = json.load(f)

    def generate_rag_text(fields):
        """Convert field definitions into human-readable text."""
        lines = []
        for field, props in fields.items():
            lines.append(f"Field: {field}")
            lines.append(f"  - Type: {props['type']}")
            lines.append(f"  - Description: {props['description']}")
            lines.append(f"  - Nullable: {props['nullable']}")
            lines.append("")
        return "\n".join(lines)

    # Dictionary to hold documents by section title
    section_documents = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for section, fields in schema.items():
        rag_text = generate_rag_text(fields)
        doc = [
            Document(page_content=rag_text, metadata={"object_type": section})
        ]
        section_documents[section] = text_splitter.split_documents(doc)

    # print(section_documents["accumulatedcores"])
    for item, doc in section_documents.items():
        # print(item)
        _ = vector_store.add_documents(doc)

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""
)

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = "If I am looking for the star formation rate over black hole radiation, what variable am I looking for?"
    print(question)
    response = graph.invoke({"question": question})
    print(response["answer"])
    

    question = "What is sfr/bhr?"
    print(question)
    response = graph.invoke({"question": question})
    print(response["answer"])


if __name__ == "__main__":
    main()
