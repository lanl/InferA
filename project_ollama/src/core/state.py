from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):    
    # Sequence of messages exchanged
    messages: Annotated[list[BaseMessage], add_messages]
    user_inputs: Annotated[list[BaseMessage], add_messages]

    session_id: str
    state_key: str

    model: str

    current: str
    next: str

    result: str

    # supervisor node
    task: str
    current_step: int
    
    # store important msgs for specific error handling use
    stashed_msg: str

    # dataloader node
    file_index: dict
    object_type: list
    current_obj: int

    # written database information
    db_path: str
    db_tables: list
    db_columns: list[list]

    # retriever node
    retrieved_docs: list

    # planner node
    plan: dict
    plan_verified: False

    # qa node
    qa_retries: int
    qa_failed: bool

    # data analysis nodes
    results_list: list
    df_index: int

