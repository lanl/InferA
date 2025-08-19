"""
Module: state.py

Purpose:
    Defines the shared conversational state structure used in LangGraph-based multi-agent workflows.
    This `State` class captures the full memory and context required for decision-making across nodes,
    including message history, current task tracking, data analysis status, QA cycles, and more.

Notes:
    - The state is passed between nodes in the LangGraph DAG.
    - All message updates are tracked using the `add_messages` update function.
    - This structure can evolve based on application complexity.

Dependencies:
    - langchain_core.messages.BaseMessage
    - langgraph.graph.message.add_messages
"""

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class State(TypedDict):    
    # === Message History ===
    messages: Annotated[list[BaseMessage], add_messages] # Full message history
    user_inputs: Annotated[list[BaseMessage], add_messages] # User-specific messages (for filtering)

    # === Session & Model Info ===
    session_id: str                       # Unique identifier for the conversation/session
    model: str                            # Identifier for the model being used (OpenAI, Ollama, etc.)

    # === Graph Navigation State ===
    current: str                          # Current LangGraph node
    next: str                             # Next LangGraph node (used for transitions)

    # === Shared Result Container ===
    result: str                           # General-purpose result (used between nodes)

    # === Supervisor Node Tracking ===
    task: str                             # Current high-level task (e.g., "generate plot")
    base_task: str                        # Persistent default task for fallback for QA fails
    current_step: int                     # Current step index in task execution

    # === Human-in-the-Loop Feedback ===
    approved: bool                        # Whether the last result was approved by a human
    approve_msg: str                      # Optional feedback message from human
    
    # === Error Recovery ===
    stashed_msg: str                      # Important message to stash for recovery after error

    # === DataLoader Node ===
    file_index: dict                      # Mapping of file names to hierarchy
    object_type: list                     # Types of loaded objects (e.g., "halo", "galaxy", "haloparticles")
    current_obj: int                      # Index of the current object being processed

    # === Database Information ===
    db_path: str                          # Path to connected duckDB database
    db_tables: list                       # List of table names in the DB
    db_columns: list[list]                # List of columns per table

    # === Retriever Node ===
    retrieved_docs: dict                  # Documents retrieved during context search

    # === Planner Node ===
    plan: dict                            # Parsed task plan from planner agent

    # === Data Analysis Results ===
    results_list: list                    # List of finalized results
    working_results: list                 # Intermediate or unapproved results (retryable)
    df_index: int                         # Index for tracking which DataFrame is in focus

    # === QA Node State ===
    qa_retries: int                       # Count of QA attempts for current task
    qa_failed: bool                       # Whether the QA node has flagged the result as failed

    # === Documentation Node ===
    last_documentation: str              # Most recent generated documentation text
