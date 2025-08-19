"""
Module: human_feedback_node.py
Purpose:
    This module defines the `HumanFeedback` node for agentic pipelines that require human oversight.
    It handles decision approval from a user, either interactively via CLI or bypassed programmatically
    using the `DISABLE_FEEDBACK` configuration.

    The node is typically called when an agent (e.g., DataLoader) needs user confirmation before proceeding.

Features:
- Interactive approval via terminal when DISABLE_FEEDBACK is False
- Auto-approval with default messages when DISABLE_FEEDBACK is True
- Handles fallback routing when human feedback is disabled or not needed

Classes:
- Node(NodeBase): Main class responsible for managing human approval in agent workflows.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.node_base import NodeBase
from config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)



class Node(NodeBase):
    """
    HumanFeedback node for handling agent interaction with human reviewers.

    This node is responsible for pausing the flow to collect approval or feedback
    from a human operator. It can operate in two modes:

    - Interactive: Asks for input in the terminal (default)
    - Programmatic: Automatically approves if DISABLE_FEEDBACK is set to True
    """
    def __init__(self):
        super().__init__("HumanFeedback")        
    
    def run(self, state):
        """
        Runs the human feedback step.

        If DISABLE_FEEDBACK is enabled:
            - Automatically approves the action and routes back to the previous node.
            - Adds default HumanMessage based on context.

        If interactive:
            - Prompts the user to approve or provide feedback.
            - Returns a HumanMessage containing the feedback.

        Args:
            state (dict): The current state of the pipeline, including:
                - current (str): The name of the node that requested feedback.
                - approved (bool, optional): Whether the last action was already approved.

        Returns:
            dict: Updated state including:
                - messages: List containing the human response
                - user_inputs: Echo of the feedback as input
                - approved (bool): Whether the feedback was positive
                - next (str): The name of the node to return to
                - current (str): The name of this node ("HumanFeedback")
        """
        previous_node = state.get("current", None)

        # Avoid feedback loop or redundant approvals
        if previous_node in ["HumanFeedback", "RoutingTool"]:
            previous_node = "Supervisor"
        approved = state.get("approved", None)
        
        logger.debug(previous_node)

        # === Mode: Auto-approve if feedback is disabled ===
        if DISABLE_FEEDBACK:            
            if previous_node in ["DataLoader"]:
                return {
                    "messages": [AIMessage("\033[1;31mSkipping human feedback in DataLoader.\033[0m")], 
                    "next": previous_node, 
                    "user_inputs": [HumanMessage("You may ignore those variables and continue forward with any variables you have available to you.")],
                    "current": "HumanFeedback",
                    "approved": True
                }
            # Generic fallback for all other nodes
            return {
                "messages": [AIMessage("\033[1;31mSkipping human feedback (automatic approval). Sending back to previous node.\033[0m")], 
                "next": previous_node, 
                "user_inputs": [HumanMessage("Approved")],
                "current": "HumanFeedback",
                "approved": True
            }
        
        # === Mode: Interactive approval ===
        logger.info(f"--- Human feedback requested ---")

        user_input = input(
            "\n\033[1m\033[31mDo you approve? Please reply with Y or Yes to approve.\n"
            "If not, please provide your feedback or suggestions for improvement.\033[0m\n"
        ).strip().lower()
        
        if user_input in ['y', 'yes']:
            feedback = HumanMessage("Approved")
            approved = True
        else:
            feedback = HumanMessage(f"Not approved.\nFeedback: {user_input}")
            approved = False
            
        logger.debug(user_input)
        logger.debug(f"[HUMAN FEEDBACK] feedback added to messages.\nFeedback: {user_input}")

        return {
            "messages": [feedback], 
            "user_inputs": [feedback], 
            "approved": approved, 
            "next": previous_node, 
            "current": "HumanFeedback"
        }