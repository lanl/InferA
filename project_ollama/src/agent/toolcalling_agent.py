from src.tools.tool_registry import filter_data

from langgraph.prebuilt import create_react_agent

def create_toolcalling_agent(llm, members):
    """Create toolcalling agent"""
    agent = create_react_agent(
        model = llm,
        tools = [load_data, filter_data],
        prompt = """
        You are a skilled Python programmer specializing in data processing and analysis, working under the guidance of a Supervisor Agent. Your role is to:

        1. Interpret the Supervisor’s instructions to determine which data processing tool to run next.
        2. Wait for further instructions or confirmation from the Supervisor before proceeding.

        Constraints:
        - Focus solely on data processing tasks. 
        - Avoid unnecessary complexity; prioritize readability and efficiency.
        - Do not decide if the overall analysis is complete; always defer to the Supervisor for final decisions.
        """,
        name = "ToolCaller",
    )
    return agent