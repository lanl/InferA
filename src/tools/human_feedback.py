"""
Module: human_feedback.py
Purpose: This module facilitates manual approval from a human-in-the-loop during agent workflows,
         typically used within LangChain graph workflows or similar pipelines.

Functions:
    - human_feedback(approve_msg): Prompts a human user for feedback or approval before continuing.
"""

import logging
from langchain_core.messages import HumanMessage
from config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)


def human_feedback(approve_msg) -> dict:
    """
    Prompts a human user to approve or reject a proposed action, allowing for manual intervention
    in automated workflows. If DISABLE_FEEDBACK is enabled, this function auto-approves.

    Args:
        approve_msg (str): A message to display explaining what is being approved.

    Returns:
        tuple:
            - HumanMessage: LangChain-compatible message containing feedback or approval
            - bool: True if approved, False otherwise
    """
    logger.info(approve_msg)
    logger.info("--- Human feedback requested ---")

    # If feedback is disabled via config, auto-approve and return
    if DISABLE_FEEDBACK:
        logger.info("Skipping human feedback (leave approval to agent).")
        feedback = HumanMessage("Approved")
        approved = True
        return feedback, approved

    # Otherwise, prompt user for approval interactively
    try:
        user_input = input(
            "\n\033[1m\033[31mDo you approve? Reply with Y or Yes to approve.\n"
            "If not, please provide your feedback or suggestions for improvement:\033[0m\n"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        logger.warning("Feedback input interrupted by user.")
        user_input = ""

    # Check if user approved
    if user_input in ['y', 'yes']:
        feedback = HumanMessage("Approved.")
        approved = True
    else:
        feedback = HumanMessage(f"Not approved.\nFeedback: {user_input}")
        approved = False
        
    logger.debug(f"[HUMAN FEEDBACK] feedback added to messages.\nFeedback: {user_input}")

    return feedback, approved