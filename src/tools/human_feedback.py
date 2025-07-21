import logging
from langchain_core.messages import HumanMessage

from config import DISABLE_FEEDBACK

logger = logging.getLogger(__name__)

def human_feedback(approve_msg) -> dict:
    """
    Requests human approval or feedback before continuing.
    Designed for integration into LangChain workflows.

    Args:
        state (dict): The current agent state. Expected keys:
                      - "current": current node identifier
                      - "approved": approval status (optional)

    Returns:
        dict: Update to the state with feedback and next node
    """     
    logger.info(approve_msg)
    
    logger.info("--- Human feedback requested ---")

    if DISABLE_FEEDBACK:
        logger.info("Skipping human feedback (leave approval to agent).")
        feedback = HumanMessage("Approved")
        approved = True
        return feedback, approved

    try:
        user_input = input(
            "\n\033[1m\033[31mDo you approve? Reply with Y or Yes to approve.\n"
            "If not, please provide your feedback or suggestions for improvement:\033[0m\n"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        logger.warning("Feedback input interrupted by user.")
        user_input = ""

    if user_input in ['y', 'yes']:
        feedback = HumanMessage("Approved.")
        approved = True
    else:
        feedback = HumanMessage(f"Not approved.\nFeedback: {user_input}")
        approved = False
        
    logger.debug(user_input)
    logger.debug(f"[HUMAN FEEDBACK] feedback added to messages.\nFeedback: {user_input}")
    return feedback, approved