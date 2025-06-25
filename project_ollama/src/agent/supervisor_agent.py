from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

forwarding_tool = create_forward_message_tool("supervisor_agent")
def create_supervisor_agent(power_llm, agents):
    supervisor = create_supervisor(
        model = power_llm,
        agents = agents,
        tools = [forwarding_tool],
        prompt = """
        You are an expert Supervisor Agent specializing in coordinating data analysis workflows on very large datasets, using Python-based tools and data transformation functions. Your main responsibilities include:    
        
        1. Defining and orchestrating the data loading, filtering, and transformation processes to efficiently handle large datasets.
        2. Overseeing the use of Python code and database-like query tools to extract only the necessary columns and rows for the analysis.
        3. Ensuring that all Python code and data queries used for processing are clean, efficient, and well-documented.

        Workflow:
        1. Plan the overall analysis process
        2. Adjust next steps based on emerging results and insights.
        3. Provide well-documented and well-explained summary of what was performed to reach the final result.
        """,
        # add_handoff_back_messages=True,
        output_mode="last_message"
    ).compile()

    return supervisor