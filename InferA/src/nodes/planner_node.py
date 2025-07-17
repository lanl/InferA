import logging
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.nodes.node_base import NodeBase

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
        
        self.system_prompt = '''
            You are a planner for a data analysis project and an expert on cosmology simulations, with a specialization in data analysis using pandas dataframes and sql queries.
            Your expertise is in formulating plans and delegating tasks to members of your team to complete tasks related to large-scale cosmology simulation data analysis.
            You provide detailed and extensive steps to the responsible team member that can be executed sequentially to solve the user's task.

            Your team members are as follows:"
            - DataLoader: This member loads the necessary files for downstream analysis from the large set of files in the database, and is aware of the file contents. This member writes those files to a database which all other agents will have access to. If visualization is required, make sure to ask DataLoader to load coordinate data too.
            - SQLProgrammer: This member filters data for large datasets. He is your go-to first agent so that the other data analysis members are not overwhelmed by large amounts of data. This agent should try to collect ALL information necessary into a database - leave filtering to PythonProgrammer. However, he only has access to basic SQL features. If the SQL query requires multiple lines, break the problem down into different SQL queries or pass that task to PythonProgrammer.
            - PythonProgrammer: This member is a world-class python programmer, and is able to perform more complex analysis on the given data that the SQLProgrammer cannot do with basic SQL, including algorithmic calculations. Do not ask Python Programmer to plot anything.
            - Visualization: This member is able to take coordinate data from the python programmer and visualize it. In order to visualize coordinates, dataloader must load coordinate columns from the database.
            - Summary: This member summarizes the final result. Always add the Summary member at the end of the task to summarize findings.

            The simulation data is outlined as follows: "
            - simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps.
            - timestep: Each timestep is a folder containing cosmology particles from that simulated timestep.
            - cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies.
            - objects: Each row in the cosmology object files indicates an object (particle or large cosmology object). Make sure that when extracting information about rows, a relevant unique identifier is part of the extracted data.
            
            Note that you are not required to delegate to all of your team members. Limit the extent of your task to what the user requested. Each step should only call one of the members and provide a summary of the tasks that member should complete.
            
            **Example:**
            TASK: Visualize the 100 largest halos and 100 largest galaxies from timestep 498 in simulation 0.
            - Step 1. DataLoader: Identifies halo and galaxy files and also writes those data files to a database.
            - Step 2. SQLProgrammer: Reads the database, runs a SQL query to find the 100 largest halos, returns a DataFrame.
            - Step 3. SQLProgrammer: Reads the database, runs a SQL query to find the 100 largest galaxies, returns a DataFrame.
            - Step 4. PythonProgrammer: Takes DataFrames from previous SQL steps, combines them into one dataframe.
            - Step 5. Visualization: Loads DataFrame from previous step, writes a VTP file plotting the coordinates of all 100 halos and galaxies.
            - Step 5. Summary: Summarizes the results from PythonProgrammer and Visualization.
            - Complete. No need to call other members for additional tasks.
            Respond only with JSON, dividing each step of the task.
        '''

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
        
    def run(self, state):
        task = state["messages"]
        approved = state.get("approved", False)
        if not approved: 
            response = self.chain.invoke({'message': task})
            return {
                "messages": [{
                    "role": "assistant", 
                    "content": f"Plan for analysis:\n{self.pretty_print_steps(response)}"
                }], 
                "plan": response, 
                "current": "Planner",
                "next": "HumanFeedback"
                }
        else:
            # Skip planning; assume existing plan is approved
            return {
                "messages": [{
                    "role": "assistant", 
                    "content": "Plan approved. Beginning analysis."
                }], 
                "next": "Supervisor",
                "current": "Planner",
                "approved": False
            }
    

    def pretty_print_steps(self, plan):
        str = f"\n\033[1;35mTask: {plan["task"]}\033[0m\n\n"
        count = 1
        for i in plan['steps']:
            str += f"\033[1;35m  - Step {count}: {i['description']}\033[0m\n\n"
            count += 1
        return str
