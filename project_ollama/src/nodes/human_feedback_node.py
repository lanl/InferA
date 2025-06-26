import logging
from langchain_core.messages import HumanMessage
from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self):
        super().__init__("HumanFeedback")        
    
    def run(self, state):
        print(f"--- Human feedback requested ---")
        feedback = HumanMessage(input("\n\033[1m\033[31mPlease provide feedback:\033[0m\n"))

        logger.info(f"[HUMAN FEEDBACK] feedback added to messages")
        return {"messages": [feedback], "user_inputs": [feedback], "next": state["current"], "current": "HumanFeedback"}