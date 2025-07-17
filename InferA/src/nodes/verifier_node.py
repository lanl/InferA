import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("Verifier")
    
    def run(self, state):
        # Based on user feedback, revise plan or continue to steps
        previous_node = state.get('current', None)
        stashed_msg = state.get("stashed_msg", "")
        approved = state.get("approved", None)
        
        if not stashed_msg:
            return {"messages": [AIMessage(f"No message queued for approval. Skipping verification.")], "next": previous_node, "current": "Verifier"}
        
        if previous_node in ["Planner", "SQLProgrammer", "PythonProgrammer", "QA"]:
            logger.info(f"[VERIFIER] Routed directly from planner. Asking for human feedback first.")       
            return {"next": "HumanFeedback", "messages": [AIMessage(f"\033[1;35mAre you satisfied with the plan? If not, you may respond with changes you would like.\033[0m")]}
        
        elif previous_node in ['HumanFeedback']:
            logger.info(f"[VERIFIER] Routed from human feedback. Check if feedback is positive or negative.")
            
            return {"messages": [response], "next": "RoutingTool", "current": "Verifier"}