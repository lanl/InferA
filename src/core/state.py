from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):    
    # Sequence of messages exchanged
    messages: Annotated[list[BaseMessage], add_messages]
    user_inputs: Annotated[list[BaseMessage], add_messages]

    session_id: str

    model: str

    current: str
    next: str

    result: str

    # supervisor node
    task: str
    base_task: str # store default task to the agent to regenerate if needed
    current_step: int

    # human feedback
    approved: bool
    approve_msg: str
    
    # store important msgs for specific error handling use
    stashed_msg: str

    # dataloader node
    file_index: dict
    object_type: list
    current_obj: int # This allows dataloader to loop through all the different objects needed to process

    # written database information
    db_path: str
    db_tables: list
    db_columns: list[list]

    # retriever node
    retrieved_docs: dict

    # planner node
    plan: dict

    # data analysis nodes
    results_list: list
    working_results: list # This is for written results that are flagged by QA agent does not get added to results_list so that it can be re-run
    df_index: int

    # qa node
    qa_retries: int
    qa_failed: bool

    # documentation node
    last_documentation: str
