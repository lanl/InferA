import logging
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase

from config import MESSAGE_HISTORY

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("Supervisor")
        self.llm = llm.bind_tools(tools, tool_choice = "redirect")
        self.system_prompt = """
            You are a supervisor coordinate a team of specialized agents to process and analyze large-scale cosmology simulation data.
            Your role is to follow a pre-written plan and delegate tasks to the appropriate team members based on their responses.

            < INSTRUCTIONS >
            1. Read the response from the previous agent carefully.
            2. Determine the next step based on the pre-written plan and the previous agent's output.
            3. Redirect to the appropriate agent for the next task using the Routing Tool.
            4. If clarification is needed, redirect to HumanFeedback.

            - Redirect ONLY one of the following per step: DataLoader, SQLProgrammer, PythonProgrammer, Visualization, HumanFeedback, Summary or END 
            - Stick strictly to the pre-written plan unless HumanFeedback suggests a change.
            - If the previous agent's response indicates an error or unexpected result, redirect to HumanFeedback for guidance and route back to previous agent.

            < TEAM MEMBERS >
            - DataLoader: Loads and manages data files.
            - SQLProgrammer: Writes and executes SQL queries and data filtering.
            - PythonProgrammer: Writes and executes python code for complex calculations and analyses.
            - Visualization: Creates visual representations of data.
            - HumanFeedback: Provides clarification or additional information.
            - Summary: Summarizes results at the end of the task.
            - END: Indicates the completion of all tasks.

            < PLAN >
            {plan}

            < Previous agent >
            {previous_member}

            < Previous agent's response (if any) >
            {previous_response}

            < User feedback (if any) >
            {user_feedback}

            Respond with the redirection and the next subtask based on the plan and the previous response.
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
        messages = state["messages"]
        plan = state["plan"]
        steps = plan['steps']

        previous_node = state.get("current", "")
        stashed_msg = state.get("stashed_msg", "")
        user_input = state.get("user_input", [])

        if previous_node in ["HumanFeedback"]:
            user_input = user_input[-1]
        else:
            user_input = ""

        # task to be sent to agents is updated in the tool along with the routing
        response = self.chain.invoke({'message': messages[-MESSAGE_HISTORY:],'plan': plan, 'previous_member': previous_node, 'previous_response': stashed_msg, 'user_feedback': user_input})
        logger.debug(f"[SUPERVISOR] {response.tool_calls}")

        return {"messages": [response], "current": "Supervisor", "next": "RoutingTool"}