"""
Module: dataloader_node.py

Purpose:
    This module defines the `Node` class for the DataLoader node in an Agentic Simulation Data Pipeline.
    It orchestrates loading simulation data files into a database by coordinating two specialized agents:
    - A loading agent to select and load files using metadata.
    - A writing agent to decide which columns of the loaded data to persist in the database.

    The node runs through various states of the data ingestion pipeline using language model chains.
    It includes fallback mechanisms such as human feedback if the agents are uncertain.

Functions / Classes:
- Node(NodeBase): Main class managing data loading, column retrieval, and DB writing decisions.
"""

import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from src.nodes.node_base import NodeBase
from src.tools.human_feedback import human_feedback
from src.utils.json_loader import open_json

from config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)


class Node(NodeBase):
    """
    DataLoader node managing simulation data loading and ingestion.

    This class coordinates two specialized agents:
    - A loading agent that identifies and loads simulation files using metadata.
    - A writing agent that selects relevant columns from loaded data for DB writing.

    Attributes:
        llm_load: Language model instance bound to loading tools.
        llm_write: Language model instance bound to DB writing tools.
        load_prompt_template: Prompt template guiding the loading agent.
        write_prompt_template: Prompt template guiding the writing agent.
        chain_load: Combines load prompt with loading llm to form a chain.
        chain_write: Combines write prompt with writing llm to form a chain.
    """
    def __init__(self, llm, load_tools, db_tools):
        """
        Initialize DataLoader node with language models and associated tools.

        Args:
            llm: Base language model instance to bind with tools.
            load_tools: List of tools for loading files (e.g., load_file_index).
            db_tools: List of tools for writing data to the database (e.g., load_to_db).

        User customization points:
        - Modify `load_prompt` to change how the loading agent interprets tasks and metadata.
        - Modify `write_prompt` to change criteria for selecting columns to write.
        - Change tools passed in `load_tools` or `db_tools` to extend functionality.
        """
        super().__init__("DataLoader")

        # Bind language models to their respective tools
        self.llm_load = llm.bind_tools(load_tools, parallel_tool_calls = False)
        self.llm_write = llm.bind_tools(db_tools, parallel_tool_calls = False)

        # Prompt guiding the loading agent on which files to load
        self.load_prompt = """
            You determine which files are necessary to load for a task.\n
            < Task >
            {task}\n\n
            
            These are the object types available and a description of each object type:
            < Objects available and descriptions >
            {context}
            
            1. Use the context below (column names) to decide what metadata and object_type is relevant.\n
            2. Call load_file_index() with the correct metadata.\n
            3. If a required parameter is missing, ask the user for those parameters. Otherwise, call the load_file_index() tool.
            4. If multiple simulations, objects, or timesteps, include all of them in a list.
            """
        self.load_prompt_template = PromptTemplate(
            template = self.load_prompt,
            input_variables=["task", "context"]
        )
        self.chain_load = self.load_prompt_template | self.llm_load

        # Prompt guiding the writing agent on which columns to write to the DB
        self.write_prompt ="""
            You are a data ingestion agent. Your job is to determine which columns from a dataset should be written to a database.

            The supervisor has given you this task:
            < Task >
            {task}

            < Object type to write to database >
            {object}

            < Description of object type>
            {description}

            < Column names and their definitions related to object type >
            {context}

            < User input >
            {user_input}

            **Your goal is to:**
            1. Based on the task and the object type, determine which columns are related to completing the task.
            2. Always include a column that functions as a unique identifier. Use the context of column names to determine which column to extract as the unique identifier.
            3. If you are uncertain which columns are relevant or you think columns are missing, do not return a tool. Instead, ask the user for clarification.
            
            Note that the data that the user asks for may not be the name of the actual column in the database.
            Call the tool a maximum of one time with one object_type.
            Do not include extra commentary. Only call the tool once you're confident in your selection.
            """
        self.write_prompt_template = PromptTemplate(
            template = self.write_prompt,
            input_variables=["task", "context", "description", "user_input"]
        )
        self.chain_write = self.write_prompt_template | self.llm_write
        
    
    def run(self, state):
        """
        Executes one step of the DataLoader node's logic based on the current state.

        Handles the main workflow:
        - If file index is missing, runs the loading chain to determine files to load.
        - If columns are missing, routes to a retriever node to fetch columns.
        - Uses the writing chain to select which columns to write to the database.
        - If all tasks completed, transitions to the Documentation node.

        Args:
            state (dict): Contains current data pipeline state with keys:
                - task (str): Current task description.
                - user_input (list or str): User responses or clarifications.
                - file_index (optional): Metadata about loaded files.
                - object_type (list): List of object types to process.
                - current_obj (int): Index tracking which object_type is currently processed.
                - approved (bool): Whether the user has approved the column selection.
                - retrieved_docs (dict): Columns or documents retrieved for each object_type.
                - db_path (str): Path indicating where data was written in DB.

        Returns:
            dict: Contains updated messages, next node/tool, current node, and tracking info.

        User customization points:
        - Change the maximum number of retries (currently 3) for approval in the write stage.
        - Customize routing behavior or add new fallback states.
        """
        task = state.get("task")
        user_input = state.get("user_input", [])
        file_index = state.get("file_index", None)
        object_type = state.get("object_type", [])
        current_obj = state.get("current_obj", 0)
        approved = state.get("approved", False)
        retrieved_docs = state.get("retrieved_docs", None)
        db_path = state.get("db_path", "")

        # Load file descriptions from JSON to provide context for the agents
        try:
            file_descriptions = open_json("src/data/file_descriptions.json")
        except Exception as e:
            logger.error(f"[DATALOADER] Failed to load file_descriptions.json: {e}")
            file_descriptions = {}
        

        # Stage 1: Determine which files to load if not yet identified
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


        # Stage 2: Route to retriever node if columns have not been retrieved yet
        if not retrieved_docs:
            if not object_type:
                logger.debug(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node. No object_type found, retrieving from all docs.") 
            else:
                logger.debug(f"[DATALOADER] Column names not retrieved yet. Routing to retriever node.")       
            return {"next": "Retriever", "current": "DataLoader", "messages": [AIMessage("Retrieving relevant columns. Sending to retriever node.")]}


        # Stage 3: Use writing agent to determine columns to persist in the DB for each object_type    
        if current_obj < len(object_type):
            obj = object_type[current_obj]
            obj_description =file_descriptions[obj]

            # Initialize user input and approval status for feedback loop
            user_input = ""
            approved = False
            counter = 0 

            # Limit retries to prevent infinite loops; can be customized by user
            while not approved or counter < 3:
                response = self.chain_write.invoke({
                    "task": task, 
                    "context": retrieved_docs[obj], 
                    "object": obj,
                    "description": obj_description,
                    "user_input": user_input,
                })
                counter += 1

                # If feedback disabled, automatically approve if tools called
                if DISABLE_FEEDBACK:
                    if not response.tool_calls:
                        user_input = "No, ignore the columns that you are missing and continue."
                    else:
                        approved = True
                        user_input = ""
                else:
                    # If no tools called, route to human feedback for clarification
                    if not response.tool_calls:
                        logger.warning(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
                        return {"messages": [response], "current_obj": current_obj, "next": "HumanFeedback", "current": "DataLoader"}
                    
                    feedback, approved = human_feedback(f"{response.content}\n{response.tool_calls}\n")
                    user_input = "\n".join([user_input, feedback.content])

                # Once approved, move to next object_type and transition to DBWriter
                if approved:
                    current_obj += 1
                    logger.debug(f"[DATALOADER] Tool called.")
                    return {"messages": [response], "current_obj": current_obj, "next": "DBWriter", "current": "DataLoader", "approved": False}
            
            
        # Stage 4: If all loading and writing completed, signal completion and move on
        if retrieved_docs and file_index and db_path:
            logger.debug(f"[DATALOADER] All dataloading tasks complete.")
            return {
                "messages": [AIMessage(f"✅ \033[1;32mAll dataloading tasks completed. Loaded in necessary data, and wrote to: \n{db_path}\033[0m")], 
                "next": "Documentation", 
                "current": "DataLoader"
            }
