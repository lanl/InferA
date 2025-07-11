import json
import logging
from typing_extensions import List, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class RetrieverNode(NodeBase):
    def __init__(self, embedding_model, schema_path="src/data/JSON/data_variables.json"):
        super().__init__("RetrieverNode")
        self.embedding_model = embedding_model
        self.schema_path = schema_path
        self.vector_store = None
        # self.vector_store = self._build_vector_store()

    def run(self, state):
        self.vector_store = self._build_vector_store()

        retriever = self.vector_store.as_retriever()
        state["retriever"] = retriever
        return state

    def _build_vector_store(self):
        with open(self.schema_path) as f:
            schema = json.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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