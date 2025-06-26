from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

# @tool(parse_docstring=True)
# def redirect(
#     next: str, tool_call_id: Annotated[str, InjectedToolCallId]
# ) -> Command:
#     """A tool that redirects to a specific agent.
    
#     Args:
#         next: Name of the agent to redirect to.
#     """
#     # Returning a text response and updated value for the state
#     return Command(update={
#         "next": next,
#         "messages": [
#             ToolMessage(
#                 f"You will be redirected to {next}",
#                 tool_call_id=tool_call_id,
#             )
#         ]
#     })

# defining a tool with `content_and_artifact` return
@tool(parse_docstring=True)
def redirect(
    next: str, task: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """A tool that redirects to a specific agent.
    
    Args:
        next: Name of the agent to redirect to.
    """
    # Returning a text response and updated value for the state
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