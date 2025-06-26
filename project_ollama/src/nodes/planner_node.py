import logging
from typing import List, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Step(TypedDict):
    description: str = Field(..., description="A single step to accomplish the task")

class FormattedPlan(TypedDict):
    task: str = Field(..., description="The overall task to be completed")
    steps: List[Step] = Field(..., description="A list of steps to accomplish the task")

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("PlannerNode")
        self.llm = llm.with_structured_output(FormattedPlan)
        
        self.system_prompt = (
            "You are a planner for a data analysis project and an expert on cosmology simulations, with a specialization in data analysis using pandas dataframes and sql queries." \
            "Your expertise is in formulating plans and delegating tasks to members of your team to complete tasks related to large-scale cosmology simulation data analysis." \
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
            "If there is more context that you need from the user, directly ask the user for additional information that you need." \
            "Note that you are not required to delegate to all of your team members. Limit the extent of your task to what the user requested." \
            "" \
            "Respond only with JSON, dividing each step of the task."
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    def run(self, state):
        task = state["messages"]
        response = self.chain.invoke({'message': task})

        return {"messages": [{"role": "assistant", "content": f"Planning next steps for analysis:\n{self.pretty_print_steps(response)}"}], 
                "plan": response, 
                "current": "Planner", 
                "current_step": 0}
    
    def pretty_print_steps(self, plan):
        str = f"\n\033[1m\033[31mTask: {plan["task"]}\033[0m\n\n"
        count = 1
        for i in plan['steps']:
            str += f"\033[1m\033[31m   - Step {count}: {i['description']}\033[0m\n\n"
            count += 1
        return str
