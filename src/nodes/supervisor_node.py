"""
Module: supervisor_node.py
Purpose:
    This module defines the Supervisor node, which acts as a central coordinator 
    in a multi-agent pipeline for analyzing large-scale cosmology simulation data.

Responsibilities:
    - Reads the current plan and latest messages.
    - Delegates tasks to appropriate agent nodes based on a structured plan.
    - Uses tool-augmented LLM prompting to determine routing decisions.
    - Routes agents in the following order: DataLoader → SQLProgrammer → PythonProgrammer 
      → Visualization → Summary → END.

Classes:
    - Node: Supervisor node responsible for task delegation and routing control.

Inputs:
    - plan: Dict representing the current analysis workflow.
    - messages: A list of previous conversation messages (chat history).
    - stashed_msg: Result or error message from the last agent's run.
    - user_input: Optional user feedback, especially if HumanFeedback was invoked.

Outputs:
    - A routing decision indicating which node should act next.
    - Preserves updated conversation and task state.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase
from config import MESSAGE_HISTORY

logger = logging.getLogger(__name__)



class Node(NodeBase):
    def __init__(self, llm, tools):
        """
        Initializes the Supervisor node.

        Args:
            llm: A language model instance capable of reasoning over plans and agent outputs.
            tools: A list of tool names (agent endpoints) for LLM-based routing.
        """
        super().__init__("Supervisor")

        # Bind tools to the LLM and default to tool_choice = "redirect"
        self.llm = llm.bind_tools(tools, tool_choice = "redirect")

        # System prompt defining Supervisor’s role and behavior
        self.system_prompt = """
            You are a supervisor coordinate a team of specialized agents to process and analyze large-scale cosmology simulation data.
            Your role is to follow a pre-written plan and delegate tasks to the appropriate team members based on their responses.

            < INSTRUCTIONS >
            1. Read the response from the previous agent carefully.
            2. Determine the next step based on the pre-written plan and the previous agent's output.
            3. Redirect to the appropriate agent for the next task using the Routing Tool.

            - Redirect ONLY one of the following per step: DataLoader, SQLProgrammer, PythonProgrammer, Visualization, Summary or END 
            - Stick strictly to the pre-written plan unless HumanFeedback suggests a change.

            < TEAM MEMBERS >
            - DataLoader: Loads and manages data files.
            - SQLProgrammer: Writes and executes SQL queries and data filtering.
            - PythonProgrammer: Writes and executes python code for complex calculations and analyses.
            - Visualization: Creates visual representations of data.
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

        # Default template (normal step)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])
        self.chain = self.prompt_template | self.llm

        # Fallback template (e.g. handling failed response message)
        self.failed_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}"),
            ("ai", "{failed_msg}")
        ])
        self.failed_chain = self.failed_template | self.llm
        
    
    def run(self, state):
        """
        Executes a Supervisor step by analyzing the latest message history, plan, and previous output,
        and returns routing information to the RoutingTool.

        Args:
            state (dict): The current pipeline state, including:
                - messages: Conversation history
                - plan: Task plan (with ordered steps)
                - current: Name of the most recent agent that acted
                - stashed_msg: Latest response or error
                - user_input: Optional user message if HumanFeedback was triggered

        Returns:
            dict: Updated state with:
                - messages: Includes the Supervisor's tool response
                - current: Set to 'Supervisor'
                - next: Always routes to 'RoutingTool' for handling the redirect
        """
        messages = state["messages"]
        plan = state["plan"]
        steps = plan['steps']

        previous_node = state.get("current", "")
        stashed_msg = state.get("stashed_msg", "")
        user_input = state.get("user_input", [])

        # If last action was HumanFeedback, extract the most recent user comment
        if previous_node in ["HumanFeedback"]:
            user_input = user_input[-1]

        # Prepare LLM input and invoke the tool routing logic
        response = self.chain.invoke({
            'message': messages[-MESSAGE_HISTORY:],
            'plan': plan, 
            'previous_member': previous_node, 
            'previous_response': stashed_msg, 
            'user_feedback': user_input
        })
        logger.debug(f"[SUPERVISOR] {response.tool_calls}")

        return {
            "messages": [response], 
            "current": "Supervisor", 
            "next": "RoutingTool"
        }