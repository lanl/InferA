import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from src.nodes.node_base import NodeBase
from src.utils.json_loader import open_json

logger = logging.getLogger(__name__)

"""
DataLoader Node for an Agentic Simulation Data Pipeline

This module defines the `Node` class responsible for orchestrating the loading 
and ingestion of simulation data files into a database within an agent-based system.

The DataLoader node interacts with two main subcomponents:
- Loading agent (llm_load): decides what metadata and file types to load
- Writing agent (llm_write): decides which dataframe columns to persist to the database

The node handles various states of the data ingestion pipeline by invoking language
model chains that utilize provided tools to load and write data, with fallbacks
for human feedback if decisions are uncertain.

Typical workflow:
- Load relevant file indices based on task and available file descriptions
- Retrieve necessary document columns
- Write selected data columns to database
- Transition to supervisor once tasks complete

Logging and error handling ensure traceability of operations.
"""

class Node(NodeBase):
    """
    DataLoader node managing simulation data loading and ingestion.

    This class coordinates two specialized agents:
    - A loading agent that identifies and loads simulation files using metadata.
    - A writing agent that selects relevant columns from loaded data for DB writing.

    Attributes:
        llm_load: Language model bound to loading tools.
        llm_write: Language model bound to database writing tools.
        load_prompt_template: Prompt template for the loading agent.
        write_prompt_template: Prompt template for the writing agent.
        chain_load: Chain that combines the loading prompt with the loading llm.
        chain_write: Chain that combines the writing prompt with the writing llm.
    """
    def __init__(self, llm, load_tools, db_tools):
        """
        Initialize DataLoader node with language models and tools.

        Args:
            llm: Base language model to bind with tools.
            load_tools: Tools for loading files (e.g., load_file_index).
            db_tools: Tools for writing to the database (e.g., load_to_db).
        """
        super().__init__("DataLoader")
        self.llm_load = llm.bind_tools(load_tools)
        self.llm_write = llm.bind_tools(db_tools)

        self.load_prompt = (
            "You are a data loading agent for simulation data. Your task is:\n"
            "{task}\n\n" \
            "" \
            "These are the object_type files available and a description of each file:" \
            "{context}" \
            ""\
            "1. Use the context below (column names) to decide what metadata and object_type is relevant.\n"
            "2. Call load_file_index with the correct metadata.\n"
            "Ask the user if any required parameter is missing.\n\n"
        )
        self.load_prompt_template = PromptTemplate(
            template = self.load_prompt,
            input_variables=["task", "context"]
        )
        self.chain_load = self.load_prompt_template | self.llm_load

        self.write_prompt = (
            "You are a data ingestion agent. Your job is to determine which columns from a dataset should be written to a database.\n\n"
            "The user has given you this task:\n"
            "{task}\n\n"
            "Here are the available files and a brief description of what is contained in each file:\n"
            "{description}\n\n"
            "Below is the context with available column names and their descriptions:\n"
            "{context}\n\n"
            ""
            "**Your goal is to:**\n"
            "1. Analyze the task.\n"
            "2. Identify columns related to the task for the object."
            "3. Always include a column that functions as a unique identifier. Use the context of column names to determine which column to extract as the unique identifier.\n"
            "4. Always incude columns that have x, y, and z coordinate data. If you do not find all three columns, ask the user for clarification."
            "5. Note that the data that the user asks for may not be the name of the actual column in the database.\n"
            "6. If you are uncertain which columns are relevant, ask the user for clarification.\n"
            "7. Do not include extra commentary. Only call the tool once you're confident in your selection."
        )

        self.write_prompt_template = PromptTemplate(
            template = self.write_prompt,
            input_variables=["task", "context", "description"]
        )
        self.chain_write = self.write_prompt_template | self.llm_write
        
    
    def run(self, state):
        """
        Run one step of the DataLoader node, processing the current state.

        The method handles multiple stages:
        - Invoking the loading chain if file indices are not yet retrieved.
        - Routing to retriever if columns are missing.
        - Invoking the writing chain to decide DB writes.
        - Finalizing when all loading tasks are complete.

        Args:
            state (dict): Dictionary containing the current context and data including:
                - messages (str): Task description/messages.
                - file_index (optional): Index of loaded files.
                - object_type (optional): Types of objects/files.
                - retrieved_docs (optional): Documented columns or data retrieved.
                - db_path (optional): Path indicating data has been written to DB.

        Returns:
            dict: A dictionary representing the next action, including keys:
                - messages: List of AIMessage or response objects.
                - next: The next node or tool to transition to.
                - current: The current node name.
        """
        task = state["messages"]
        file_index = state.get("file_index", None)
        object_type = state.get("object_type", None)
        current_obj = state.get("current_obj", 0)

        retrieved_docs = state.get("retrieved_docs", None)
        db_path = state.get("db_path", None)

        # Load file descriptions from JSON for context
        try:
            file_descriptions = open_json("src/data/file_descriptions.json")
        except Exception as e:
            logger.error(f"[DATALOADER] Failed to load file_descriptions.json: {e}")
            file_descriptions = {}
        
        # If no file index yet, run loading chain to identify files to load
        if not file_index:
            response = self.chain_load.invoke({
                "task": task, 
                "context": file_descriptions
            })

            # If no tools were called, fallback to human feedback for more info
            if not response.tool_calls:
                logger.warning(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
                return {"messages": [response], "next": "HumanFeedback", "current": "DataLoader"}
            else:
                logger.debug(f"[DATALOADER] Tool called.")
                return {"messages": [response], "next": "DataLoaderTool", "current": "DataLoader"}

        # If documents not retrieved yet, route to retriever node
        if not retrieved_docs:
            if not object_type:
                logger.debug(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node. No object_type found, retrieving from all docs.") 
            else:
                logger.debug(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node.")       
            return {"next": "Retriever", "current": "DataLoader", "messages": [AIMessage("Retrieving relevant columns. Sending to retriever node.")]}
        
        # If columns retrieved and file index present but no DB path yet, run writing chain        
        if current_obj < len(object_type):
            obj = object_type[current_obj]
            obj_description = f"**File**: {obj}\n**Description**: {file_descriptions[obj]}"

            response = self.chain_write.invoke({
                "task": task, 
                "context": retrieved_docs, 
                "description": obj_description
            })

            # If writing agent called no tools, route to human feedback
            if not response.tool_calls:
                logger.warning(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
                return {"messages": [response], "current_obj": current_obj, "next": "HumanFeedback", "current": "DataLoader"}
            else:
                current_obj += 1
                logger.debug(f"[DATALOADER] Tool called.")
                return {"messages": [response], "current_obj": current_obj, "next": "DBWriter", "current": "DataLoader"}
        
        # If all required steps completed, signal success and move to Supervisor
        if retrieved_docs and file_index and db_path:
            logger.debug(f"[DATALOADER] All dataloading tasks complete.")
            return {
                "messages": [AIMessage(f"✅ \033[1;32mAll dataloading tasks completed. Loaded in necessary data, and wrote to: \n{db_path}\033[0m")], 
                "next": "Supervisor", 
                "current": "DataLoader"
            }
