import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, load_tools, db_tools):
        super().__init__("DataLoader")
        self.llm_load = llm.bind_tools(load_tools)
        self.llm_write = llm.bind_tools(db_tools)

        self.load_prompt = (
            "You are a data loading agent for simulation data. Your task is:\n"
            "{task}\n\n"
            "1. Use the context below (column names) to decide what metadata and object_type is relevant.\n"
            "2. Call load_file_index with the correct metadata.\n"
            "Ask the user if any required parameter is missing.\n\n"
        )
        self.load_prompt_template = PromptTemplate(
            template = self.load_prompt,
            input_variables=["task"]
        )
        self.chain_load = self.load_prompt_template | self.llm_load

        self.write_prompt = (
            "You are a data ingestion agent. Your job is to determine which columns from a dataset should be written to a database.\n\n"
            "The user has given you this task:\n"
            "{task}\n\n"
            "Below is the context with available column names and their descriptions:\n"
            "{context}\n\n"
            "Your goal is to:\n"
            "1. Analyze the task.\n"
            "2. Select only the columns that are relevant to the task.\n"
            "3. Call the `load_to_db` tool using a list of column names as the only argument.\n\n"
            "Always include a column that functions as a unique identifier.. Use context of column names to determine which column to extract as the unique identifier.\n"
            "If you are uncertain which columns are relevant, ask the user for clarification.\n"
            "Do not include extra commentary. Only call the tool once you're confident in your selection."
        )

        self.write_prompt_template = PromptTemplate(
            template = self.write_prompt,
            input_variables=["task", "context"]
        )
        self.chain_write = self.write_prompt_template | self.llm_write
        
    
    def run(self, state):
        task = state["messages"]
        file_index = state.get("file_index", None)
        object_type = state.get("object_type", None)

        retrieved_docs = state.get("retrieved_docs", None)
        db_path = state.get("db_path", None)
        
        if not file_index:
            response = self.chain_load.invoke({"task": task})
            if not response.tool_calls:
                logger.info(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
                return {"messages": [response], "next": "HumanFeedback", "current": "DataLoader"}
            else:
                logger.info(f"[DATALOADER] Tool called.")
                return {"messages": [response], "next": "DataLoaderTool", "current": "DataLoader"}

        if not retrieved_docs:
            if not object_type:
                logger.info(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node. No object_type found, retrieving from all docs.") 
            else:
                logger.info(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node.")       
            return {"next": "Retriever", "current": "DataLoader", "messages": [AIMessage("Retrieving relevant columns. Sending to retriever node.")]}
        
        if retrieved_docs and file_index and not db_path:
            response = self.chain_write.invoke({"task": task, "context": retrieved_docs})
            if not response.tool_calls:
                logger.info(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
                return {"messages": [response], "next": "HumanFeedback", "current": "DataLoader"}
            else:
                logger.info(f"[DATALOADER] Tool called.")
                return {"messages": [response], "next": "DBWriter", "current": "DataLoader"}
        
        if retrieved_docs and file_index and db_path:
            logger.info(f"[DATALOADER] All dataloading tasks complete.")
            return {
                "messages": [AIMessage("All dataloading tasks completed. Loaded in necessary data, and wrote to {db_path}. SUCCESS: Move on to the next task.")], 
                "next": "Supervisor", 
                "current": "DataLoader"
            }
