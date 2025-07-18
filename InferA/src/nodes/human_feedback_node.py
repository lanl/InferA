import logging
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)

from src.nodes.node_base import NodeBase
from src.utils.config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self):
        super().__init__("HumanFeedback")        
    
    def run(self, state):
        previous_node = state.get("current", None)
        if previous_node in ["HumanFeedback"]:
            previous_node = "Supervisor"
        approved = state.get("approved", None)
        
        logger.debug(previous_node)

        if DISABLE_FEEDBACK:            
            if previous_node in ["DataLoader"]:
                return {
                    "messages": [AIMessage("\033[1;31mSkipping human feedback in DataLoader.\033[0m")], 
                    "next": previous_node, 
                    "user_inputs": [HumanMessage("You may ignore those variables and continue forward with any variables you have available to you.")],
                    "current": "HumanFeedback",
                    "approved": True
                }
            return {
                "messages": [AIMessage("\033[1;31mSkipping human feedback (automatic approval). Sending back to previous node.\033[0m")], 
                "next": previous_node, 
                "user_inputs": [HumanMessage("Approved")],
                "current": "HumanFeedback",
                "approved": True
            }
            
        logger.info(f"--- Human feedback requested ---")
        user_input = input("\n\033[1m\033[31mDo you approve? Please reply with Y or Yes to approve.\nIf not, please provide your feedback or suggestions for improvement.\033[0m\n").strip().lower()
        
        if user_input in ['y', 'yes']:
            feedback = HumanMessage("Approved")
            approved = True
        else:
            feedback = HumanMessage(f"Not approved.\nFeedback: {user_input}")
            approved = False
            
        logger.debug(user_input)
        logger.debug(f"[HUMAN FEEDBACK] feedback added to messages.\nFeedback: {user_input}")
        return {"messages": [feedback], "user_inputs": [feedback], "approved": approved, "next": previous_node, "current": "HumanFeedback"}