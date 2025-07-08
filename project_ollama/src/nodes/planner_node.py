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
            "You provide detailed and extensive steps to the responsible team member that can be executed sequentially to solve the user's task." \
            "" \
            "Your team members are as follows:"
            "- DataLoader: This member loads the necessary files for downstream analysis from the large set of files in the database, and is aware of the file contents. This member writes those files to a database which all other agents will have access to. If visualization is required, make sure to ask DataLoader to load coordinate data too."
            "- SQLProgrammer: This member filters data for large datasets. He is your go-to first agent so that the other data analysis members are not overwhelmed by large amounts of data. However, he is only able to perform one basic SQL query at a time. If the SQL requires more than 2 layers of filtering, break the problem down into different tasks."
            "- PythonProgrammer: This member is a world-class python programmer, and is able to perform more complex analysis on the given data that the SQLProgrammer cannot do with basic SQL, including algorithmic calculations."
            "- Visualization: This member is able to take the filtered data from the python programmer and visualize it." \
            "- Summary: This member summarizes the final result. Always add the Summary member at the end of the task to summarize findings."
            "" \
            "The simulation data is outlined as follows: "
            "- simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps." \
            "- timestep: Each timestep is a folder containing cosmology particles from that simulated timestep." \
            "- cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies." \
            "- objects: Each row in the cosmology object files indicates an object (particle or large cosmology object). Make sure that when extracting information about rows, a relevant unique identifier is part of the extracted data."
            "" \
            "Note that you are not required to delegate to all of your team members. Limit the extent of your task to what the user requested. Each step should only call one of the members and provide a summary of the tasks that member should complete.\n\n" \
            "Examples: TASK: Find the largest halo from timestep 498 in simulation 0.\n"
            "- DataLoader: Identifies halo files and also writes those data files to a database."
            "- SQLProgrammer: Reads the database, runs a SQL query based on schema context, which filters and extracts the largest halos from timestep 498 in simulation 0, writes data to a dataframe." \
            "- PythonProgrammer: Computes further filtering on the dataframe from SQLProgrammer." \
            "- Summary: Summarizes the results from PythonProgrammer."
            "- Complete. No need to call other members for additional tasks." \
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
                "current": "Planner"
                }
    
    def pretty_print_steps(self, plan):
        str = f"\n\033[1;35mTask: {plan["task"]}\033[0m\n\n"
        count = 1
        for i in plan['steps']:
            str += f"\033[1;35m  - Step {count}: {i['description']}\033[0m\n\n"
            count += 1
        return str
