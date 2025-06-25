from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

class State(TypedDict):    
    # Sequence of messages exchanged
    messages: Sequence[BaseMessage]

    
    # toolcaller_state: str = ""

    # supervisor_state: str = ""
    # supervisor_decision: str = ""

    # sender: str = ""


