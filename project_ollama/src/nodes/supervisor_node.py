import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("Supervisor")
        self.llm = llm.bind_tools(tools)
        self.system_prompt = (
            "You are a supervisor for a data analysis project and an expert on cosmology simulations, with a specialization in data analysis using pandas dataframes and sql queries." \
            "Your expertise is in delegating specific tasks to members of your team and reviewing their work to complete tasks related to large-scale cosmology simulation data analysis." \
            "You provide detailed steps and the responsible team member that can be executed sequentially to solve the user's task." \
            "" \
            "Your team members are as follows:"
            "- SQL Query programmer: This member filters data for large datasets. He is capable of identifying relevant columns and rows in the data and extracting only those information so that the large data is manageable. He is your go-to first agent so that the other data analysis members are not overwhelmed by large amounts of data."
            "- Python programmer: This member is a world-class python programmer, and is able to perform complex analysis on the given data including algorithmic calculations."
            "- Visualization expert: This member is able to take the filtered data from the python programmer and visualize it." \
            "" \
            "The simulation data is outlined as follows: "
            "- simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps." \
            "- timestep: Each timestep is a folder containing cosmology particles from that simulated timestep." \
            "- cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies." \
            "" \
            "If there is more context that you need from the user, directly ask the user for additional information that you need."
            ""
            "You have access to the redirect tool. You are given steps of a task - please redirect the step to the appropriate team member, including their task."
            "" \
            "Redirect ONLY to either 'SQLProgrammer','PythonProgrammer', or 'Visualization'." \
            "If the task requires multiple basic functions, please break the task down into smaller pieces." \
            "Respond with one sentence." \
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    
    def run(self, state):        
        plan = state["plan"]
        steps = plan['steps']
        current_step = state["current_step"]
        response = self.chain.invoke({'message': steps[current_step]["description"]})

        return {"messages": [response], "current": "Supervisor", "next": "RoutingTool", "current_step": current_step + 1}