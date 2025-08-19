"""
Module: retriever_node.py
Purpose: This module defines a Retriever Node that interfaces with a vector store (FAISS)
         to return relevant schema-based documents based on task inputs.

It uses embeddings to index and retrieve documentation for scientific data schemas,
helping downstream nodes (e.g., data loaders or QA evaluators) understand the context
around dataset variables.

Communicates only with the Dataloader node in this workflow.

Classes:
    - Node: Inherits from NodeBase and handles document retrieval, vector store setup,
            FAISS persistence, and server consistency checking.

Functions:
    - run(state): Performs the document retrieval based on user inputs and task context.
    - _build_vector_store(): Rebuilds the FAISS store from the local JSON schema.
    - initialize_vector_store(): Initializes or rebuilds FAISS vector store if needed.
    - write_server_info(): Persists server and embedding model info for reuse validation.
    - read_server_info(): Loads metadata to verify FAISS cache compatibility.
"""

import os
import json
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage

from src.nodes.node_base import NodeBase

logger = logging.getLogger(__name__)



class Node(NodeBase):
    """
    Retriever Node that returns documentation about data schema variables relevant to a user query.
    
    This class embeds schema definitions and uses a FAISS vector store to return
    the most relevant columns or variables to support data analysis tasks.

    Attributes:
        embedding_model: Embedding model instance used to generate vector representations.
        server: Identifier of the current server (used for cache validation).
        schema_path: Path to the JSON schema with field definitions.
        vector_store_path: Path where FAISS index and server info are stored.
    """
    def __init__(self, embedding_model, server, schema_path="src/data/JSON/data_variables.json", vector_store_path = "src/data/vector_store"):
        super().__init__("Retriever")
        logger.info("[RETRIEVER] Setting up embedded model...")

        self.embedding_model = embedding_model
        self.server = server
        self.schema_path = schema_path
        self.vector_store_path = vector_store_path

        # Vector store is either loaded from disk or rebuilt as needed
        self.vector_store = self.initialize_vector_store()

        # Change the search strategy or top-k results here
        self.retriever = self.vector_store.as_retriever(
            search_type = "mmr",                # Alternatives: "similarity", "similarity_score_threshold"
            search_kwargs = {'k': 20}           # Customize number of results returned
            # search_type = "similarity_score_threshold",
        )
        
        logger.info("[RETRIEVER] Embedded model setup.")


    def run(self, state):
        """
        Retrieves documents relevant to a task and user query by embedding the input and searching FAISS.

        Args:
            state (dict): Contains the current task, user input, and object type for filtering.

        Returns:
            dict: Response containing retrieved documents and a message for the next node.
        """
        task = state["task"]
        user_inputs = state["user_inputs"][0].content
        object_type = state.get("object_type", None)
        plan = state["plan"]

        all_docs = {}
        total_docs = 0

        # Composite query from multiple task components
        query = [
            "[IMPORTANT] Retrieve names marked as important.", 
            f"{user_inputs}", 
            f"{task}", 
            f"{plan}"
        ]

        # If object_type is specified, use it as a metadata filter for retrieval
        if object_type and isinstance(object_type, list):
            for obj in object_type:
                docs = []
                for q in query:
                    doc = self.retriever.invoke(q, filter = {"object_type": obj})
                    docs.extend(doc)
                total_docs += len(docs)
                all_docs[obj] = docs
        else:
            docs = []
            for q in query:
                doc = self.retriever.invoke(q)
                docs.extend(doc)
            total_docs += len(docs)
            all_docs["All"] = docs 

        return {
            "messages": [AIMessage(f"Retrieved {total_docs} relevant documents.\n\n{query}.")], 
            "retrieved_docs": all_docs, 
            "next": state["current"], 
            "current": "Retriever"
        }


    def _build_vector_store(self):
        """
        Constructs a FAISS vector store from schema fields defined in a JSON file.

        Returns:
            FAISS: A new FAISS index populated with schema documents.
        """
        with open(self.schema_path) as f:
            schema = json.load(f)

        # Optional: Use a text splitter if your schema entries are long
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=125, chunk_overlap=25)

        all_docs = []
        for section, fields in schema.items():
            for field, props in fields["columns"].items():
                text = f"{field} - Definition : {props['description']}".strip()
                doc = Document(page_content=text, metadata={"object_type": section, "column": field})
                all_docs.append(doc)
        
        return FAISS.from_documents(all_docs, self.embedding_model)


    def _generate_rag_text(self, fields):
        """
        Generates a combined text blob from field descriptions for RAG-style summarization.

        Args:
            fields (dict): Dictionary of field metadata.

        Returns:
            str: Concatenated field descriptions.
        """
        return "\n".join(
            f"< Column > {field} - < Description > {props['description']}\n"
            for field, props in fields.items()
        )
    
    def initialize_vector_store(self):
        """
        Initializes or rebuilds the FAISS vector store depending on version compatibility.

        Returns:
            FAISS: A usable vector store instance.
        """
        vector_store = None
        rebuild_required = False

        if os.path.exists(self.vector_store_path):
            logger.info(f"[RETRIEVER] Vector store directory found at {self.vector_store_path}")
            stored_server = self.read_server_info()

            # Validate the stored server and model to decide if rebuild is needed
            if stored_server is None:
                logger.info("⚠️ \033[1;33mNo server info found. Rebuilding FAISS model for safety.\033[0m")
                rebuild_required = True
            elif stored_server.get("server") != self.server:
                logger.info("❌ \033[1;31mStored server is not the same as current server. Rebuilding FAISS model.\033[0m")
                rebuild_required = True
            elif stored_server.get("embedding_model") != self.embedding_model.__class__.__name__:
                logger.info("❌ \033[1;31mStored embedding model is different from current model. Rebuilding FAISS model.\033[0m")
                rebuild_required = True
            else:
                try:
                    vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
                    logger.info("✅ \033[1;32mLoaded existing FAISS model successfully.\033[0m")
                except Exception as e:
                    logger.error(f"Error loading existing vector store: {str(e)}")
                    logger.info("⚠️ \033[1;33mRebuilding FAISS model due to loading error.\033[0m")
                    rebuild_required = True

        if vector_store is None or rebuild_required:
            logger.info("[RETRIEVER] Building a new vector store...")
            vector_store = self._build_vector_store()
            vector_store.save_local(self.vector_store_path)
            self.write_server_info(self.server, self.embedding_model)
            logger.info("✅ \033[1;32mNew FAISS model built and saved.\033[0m")

        return vector_store
    
    def write_server_info(self, server, embedding_model):
        """
        Writes the current server and embedding model to disk for future validation.

        Args:
            server (str): Server name or ID.
            embedding_model (BaseEmbeddingModel): The model instance.
        """
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        server_info_path = os.path.join(self.vector_store_path, "server_info.json")
        server_info = {
            "server": server,
            "embedding_model": embedding_model.__class__.__name__
        }

        with open(server_info_path, "w") as f:
            json.dump(server_info, f)

    def read_server_info(self):
        """
        Reads the persisted server and embedding model information.

        Returns:
            dict or None: Server info dictionary, or None if missing or invalid.
        """
        server_info_path = os.path.join(self.vector_store_path, "server_info.json")
        try:
            if os.path.exists(server_info_path):
                with open(server_info_path, "r") as f:
                    return json.load(f)
            return None
        except:
            return None
