import logging
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.node_base import NodeBase
from src.utils.config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self):
        super().__init__("HumanFeedback")        
    
    def run(self, state):
        previous_node = state.get("current", None)
        logger.debug(previous_node)
        if DISABLE_FEEDBACK:
            if previous_node in ["Verifier"]:
                return {"messages": [AIMessage("\033[1;31mSkipping human feedback. Sending directly to supervisor.\033[0m")], "next": "Supervisor", "current": "Verifier"}
            elif previous_node in ["DataLoader"]:
                return {"messages": [AIMessage("\033[1;31mSkipping human feedback. DataLoader cannot complete. Ending analysis.\033[0m")], "next": "END", "current": "Verifier"}
            else:
                return {"messages": [AIMessage("\033[1;31mSkipping human feedback. Sending back to previous node.\033[0m")], "next": previous_node, "current": "HumanFeedback"}
            
        logger.info(f"--- Human feedback requested ---")
        user_input = input("\n\033[1m\033[31mPlease provide feedback:\033[0m\n")
        feedback = HumanMessage(user_input)
        logger.debug(user_input)
        logger.debug(f"[HUMAN FEEDBACK] feedback added to messages")
        return {"messages": [feedback], "user_inputs": [feedback], "next": previous_node, "current": "HumanFeedback"}