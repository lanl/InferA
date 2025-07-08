import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("Supervisor")
        self.llm = llm.bind_tools(tools)
        self.system_prompt = """
            As a supervisor for a data analysis project and an expert in cosmology simulations, your role is to coordinate a team of specialized agents to process and analyze large-scale cosmology simulation data using tools like pandas and SQL.

            ### INSTRUCTIONS:
            You are tasked with delegating specific, logically ordered tasks to team members. For each step, determine which agent should handle it next and describe their subtask clearly. Use the redirect tool to pass the task along. If additional information is needed before proceeding, redirect to HumanFeedback.

            - Redirect ONLY one of the following per step: DataLoader, SQLProgrammer, PythonProgrammer, Visualization.  
            - Use HumanFeedback ONLY to gather missing or ambiguous input.  
            - Break complex workflows into small, sequential steps.  
            - Respond with only one sentence per step that includes the redirection and subtask.

            ### CONTEXT:
            Your team members are:

            - DataLoader: Loads required files from the simulation database and writes them to a shared database. Understands contents of files.
            - SQLProgrammer: Extracts and filters relevant data to reduce dataset size and highlight useful columns/rows.
            - PythonProgrammer: Conducts more complex calculations and algorithmic data analyses than SQLProgrammer using Python.
            - Visualization: Creates clear, insightful visual representations of the analyzed data.
            - HumanFeedback: Requests more information from the user if task details are unclear or missing.
            - Summary: Summarizes results at the end of the task.

            ### SIMULATION STRUCTURE:
            - simulation/: Root directory with simulations under varied initial conditions.
            - timestep/: Contains data snapshots at each calculated timestep.
            - cosmology object files: Hold spatial and property data on dark matter halos, galaxies, and their particles.

            Use your judgment to direct the analysis process efficiently from raw data to visual insight.

            ### PLAN:
            {plan}

            You are currently at step {current_step} of the plan.
            """

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm

        self.failed_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}"),
            ("ai", "{failed_msg}")
        ])

        self.failed_chain = self.failed_template | self.llm
        
    
    def run(self, state):
        plan = state["plan"]
        steps = plan['steps']
        current_step = state.get("current_step", 1)
        stashed_msg = state.get("stashed_msg", None)
        qa_failed = state.get("qa_failed", False)

        if qa_failed:
            current_step = current_step - 1
            logger.info(f"[SUPERVISOR] Restarting last step: {current_step}...")

            # Use failed_chain instead of chain
            response = self.failed_chain.invoke({
                'message': steps[current_step-1]["description"],
                'failed_msg': stashed_msg,
                'current_step': current_step,
                'plan': plan
            })
            current_step += 1
            return {"messages": [response], "current": "Supervisor", "next": "RoutingTool", "current_step": current_step, "qa_failed": False}
        
        if current_step > len(steps):
            return {"messages": [AIMessage("Completed analysis.")], "current": "Supervisor", "next": "END"}
        
        logger.info(f"[SUPERVISOR] Starting step {current_step}...\n    - TASK: {steps[current_step-1]["description"]}\n")
        print(f"\033[1;36m[SUPERVISOR] Starting step {current_step}...\n    - TASK: {steps[current_step-1]["description"]}\n\033[0m")

        # task to be sent to agents is updated in the tool along with the routing
        response = self.chain.invoke({'message': steps[current_step-1]["description"], 'current_step': current_step, 'plan': plan})

        current_step += 1
        return {"messages": [response], "current": "Supervisor", "next": "RoutingTool", "current_step": current_step}