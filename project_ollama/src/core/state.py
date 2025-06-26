from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):    
    # Sequence of messages exchanged
    messages: Annotated[list[BaseMessage], add_messages]
    user_inputs: Annotated[list[BaseMessage], add_messages]

    session_id: str

    current: str
    next: str

    code: str

    # dataloader node
    file_index: dict

    # planner node
    plan: dict
    plan_verified: False

    # supervisor node
    task: str
    current_step: int

