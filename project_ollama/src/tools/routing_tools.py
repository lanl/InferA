from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

@tool(parse_docstring=True)
def redirect(
    next: str, task: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Use this tool to **redirect the current task** to one of the specialized agents. 
    This is the ONLY mechanism to delegate work to an agent in this system. 
    You must use this tool to delegate each decision, and only one agent should be selected per step.
    
    Args:
        next: Name of the agent to redirect to.
        task: A concise and clear description of what the selected agent should do next. This is provided by the Planner agent's plan.
    """
    return Command(update={
        "next": next,
        "task": task,
        "messages": [
            ToolMessage(
                f"You will be redirected to {next}. {task}",
                tool_call_id=tool_call_id,
            )
        ]
    })