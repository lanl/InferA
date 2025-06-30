import json
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, embedding_model, schema_path="src/data/JSON/data_variables.json"):
        super().__init__("Retriever")
        logger.info("[RETRIEVER] Setting up embedded model...")
        self.embedding_model = embedding_model
        self.schema_path = schema_path
        self.vector_store = self._build_vector_store()
        self.retriever = self.vector_store.as_retriever()
        logger.info("[RETRIEVER] Embedded model setup.")


    def run(self, state):
        task = state["task"]
        user_inputs = state["user_inputs"][0].content

        query = f"User input: {user_inputs}\nTask: {task}"
        docs = self.retriever.invoke(query)
        return {"messages": [AIMessage(f"Retrieved relevant documents for query: {query}. Docs retrieved: {len(docs)}.")], "retrieved_docs": docs, "next": state["current"], "current": "Retriever"}


    def _build_vector_store(self):
        with open(self.schema_path) as f:
            schema = json.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
        vector_store = InMemoryVectorStore(self.embedding_model)

        for section, fields in schema.items():
            rag_text = self._generate_rag_text(fields["columns"])
            docs = text_splitter.split_documents([
                Document(page_content=rag_text, metadata={"object_type": section})
            ])
            _ = vector_store.add_documents(docs)

        return vector_store

    def _generate_rag_text(self, fields):
        return "\n".join(
            f"Field: {field}\n - Description: {props['description']}\n"
            for field, props in fields.items()
        )