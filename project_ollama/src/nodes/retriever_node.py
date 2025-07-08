import os
import json
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, embedding_model, server, schema_path="src/data/JSON/data_variables.json", vector_store_path = "src/data/vector_store"):
        super().__init__("Retriever")
        logger.info("[RETRIEVER] Setting up embedded model...")
        self.embedding_model = embedding_model
        self.server = server
        self.schema_path = schema_path
        self.vector_store_path = vector_store_path

        if os.path.exists(vector_store_path):
            logger.info(f"[RETRIEVER] Loading existing vector store from {vector_store_path}...")
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization = True)
        else:
            logger.info("[RETRIEVER] No existing vector store found, building a new one...")
            self.vector_store = self._build_vector_store()
            self.vector_store.save_local(vector_store_path)
            
        self.retriever = self.vector_store.as_retriever()
        logger.info("[RETRIEVER] Embedded model setup.")


    def run(self, state):
        self.check_compatible_embedding(state)

        task = state["task"]
        user_inputs = state["user_inputs"][0].content
        object_type = state.get("object_type", None)

        query = f"User input: {user_inputs}\nTask: {task}"
        all_docs = []

        if object_type and isinstance(object_type, list):
            for obj in object_type:
                docs = self.retriever.invoke(query, filter = {"object_type": obj})
                all_docs.extend(docs)
        else:
            all_docs = self.retriever.invoke(query)

        return {"messages": [AIMessage(f"Retrieved {len(all_docs)} relevant documents. {query}.")], "retrieved_docs": all_docs, "next": state["current"], "current": "Retriever", "server": self.server}


    def _build_vector_store(self):
        with open(self.schema_path) as f:
            schema = json.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        all_docs = []

        for section, fields in schema.items():
            rag_text = self._generate_rag_text(fields["columns"])
            docs = text_splitter.split_documents([
                Document(page_content=rag_text, metadata={"object_type": section})
            ])
            all_docs.extend(docs)
        
        return FAISS.from_documents(all_docs, self.embedding_model)

    def _generate_rag_text(self, fields):
        return "\n".join(
            f"Field: {field}\n - Description: {props['description']}\n"
            for field, props in fields.items()
        )
    
    def check_compatible_embedding(self, state):
        prev_server = state.get("server", None)
        if prev_server and prev_server != self.server:
            logger.info("❌ \033[1;31mPrevious server is not the same as current server. Loaded FAISS model may not be correct. Rebuilding FAISS model.\033[0m")
            os.rmdir(self.vector_store_path)
            self.vector_store = self._build_vector_store()
            self.vector_store.save_local(self.vector_store_path)
        else:
            logger.info("✅ \033[1;32mChecked FAISS model compatibility.\033[0m")
