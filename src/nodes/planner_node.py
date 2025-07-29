import logging
from typing import List, TypedDict
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate

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
            You are a sophisticated planning agent for a data analysis project and an expert on cosmology simulations, with a specialization in data analysis using pandas dataframes and sql queries.
            Your expertise is in formulating plans and delegating tasks to members of your team to complete tasks related to large-scale cosmology simulation data analysis.
            You provide detailed and fine-grained steps to the responsible team member that can be executed sequentially to solve the user's task. 

            < Members of your team >
            - DataLoader: This member loads the necessary files for downstream analysis from the large set of files in the database, and is aware of the file contents. 
                This member writes those files to a database which all other agents will have access to. 
                If visualization is required, make sure to ask DataLoader to load coordinate data too.
                If the task requires multiple timesteps and it is unclear how many, be naive and ask it to load all timesteps.

            - SQLProgrammer: This member filters data for large datasets. 
                After DataLoader has loaded the data, SQLProgrammer performs initial filtering to reduce the data volume and exports the filtered data as a CSV file. 
                This agent should collect ALL information necessary into this CSV - complex filtering will be handled by PythonProgrammer.

            - PythonProgrammer: This member is a world-class python programmer who performs two main functions: 
                (1) initial additional filtering on the CSV provided by SQLProgrammer, and 
                (2) complex analysis and computations on the filtered data. The Python Programmer prepares data that will be immediately visualized by the Visualization agent.

            - Visualization: This member takes data directly from the Python Programmer and visualizes it. 
                Each computation by the Python Programmer should be followed by visualization of those specific results before moving to the next computation.

            - Summary: This member summarizes the final result. Always add the Summary member at the end of the task to summarize findings.

            < Workflow Pattern >
            1. Start with DataLoader to load necessary data into the database
            2. Use SQLProgrammer to perform initial filtering and create a CSV file
            3. Use PythonProgrammer for any additional filtering needed on the CSV
            4. Then enter a computation-visualization loop:
            - PythonProgrammer performs a specific computation
            - Visualization agent immediately visualizes that result
            - Repeat for each distinct computation/visualization needed
            5. End with Summary agent to summarize all findings

            < Simulation data details >
            - simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps.
            - timestep: Each timestep is a folder containing cosmology particles from that simulated timestep.
            - cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies.
            - objects: Each row in the cosmology object files indicates an object (particle or large cosmology object). Make sure that when extracting information about rows, a relevant unique identifier is part of the extracted data.

            Note that you are not required to delegate to all of your team members. Limit the extent of your task to what the user requested. Each step should only call one of the members and provide a summary of the tasks that member should complete.

            < Instructions >
            1. Breakdown the task into as many fine-grained steps as possible, especially for complex tasks.
            2. Be as descriptive of the necessary step to be taken by the member as possible.
            3. Follow the workflow pattern described above for all data analysis tasks.
            4. Note that if the question is actually several tasks, you may iterate the entire workflow from DataLoader to Summary but for different tasks.
            5. Do not include information about input or output.

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
                "next": "Documentation",
                "current": "Planner",
                "approved": False
            }
    

    def pretty_print_steps(self, plan):
        str = f"\n\033[1;35mTask: {plan["task"]}\033[0m\n\n"
        count = 1
        for i in plan['steps']:
            str += f"\033[1;35m  - Step {count}: {i['description']}\033[0m\n\n"
            count += 1
        logger.info(str)
        return str
