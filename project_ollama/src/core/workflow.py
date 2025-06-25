"""
graph_builder.py

This module defines the build_graph function, which constructs a StateGraph
for an analysis workflow using various nodes for data preprocessing, user interaction,
task execution, visualization, and data exploration.

The graph represents a workflow that includes:
1. Data preprocessing
2. User input and parameter extraction
3. Task selection and execution
4. Visualization
5. Data exploration using a Pandas agent

The graph structure allows for conditional branching and looping based on the state
of the analysis and user inputs.

Usage:
    graph = build_graph(llm, embedding)

Where:
    llm: Language model for natural language processing tasks
    embedding: Embedding model for text vectorization
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.core.state import State
from src.agent.supervisor_agent import create_supervisor_agent
from src.agent.toolcalling_agent import create_toolcalling_agent

class WorkflowManager:
    def __init__(self, language_models, working_directory):
        """Initialize workflow manager for analysis workflow.
        
        Args:
            language_models (dict): Dictionary of all language models - llm, power_llm, json_llm, embed
            working_directory (str): Path to the working directory
        """
        self.language_models = language_models
        self.working_directory = working_directory
        self.workflow = None
        self.memory = None
        self.graph = None
        self.members = ["toolcalling_agent"]
        self.agents = self.create_agents()
        self.setup_workflow()


    def create_agents(self):
        """Create all the agents used by the workflow"""
        llm = self.language_models["llm"]
        power_llm = self.language_models["power_llm"]
        json_llm = self.language_models["json_llm"]

        # Dictionary of agents
        agents = {}
        # Create agents using different LLMs depending on their function
        agents["toolcalling_agent"] = create_toolcalling_agent(
            llm,
            self.members,
        )

        agents["supervisor_agent"] = create_supervisor_agent(power_llm, list(agents.values()))

        return agents
    

    def setup_workflow(self):
        self.workflow = StateGraph(State)
        """Setup langgraph graph as workflow"""

        # Add nodes
        self.workflow.add_node("Supervisor", self.agents["supervisor_agent"], destinations=("ToolCaller", END))
        self.workflow.add_node("ToolCaller", self.agents["toolcalling_agent"])

        self.workflow.add_edge(START, "Supervisor")
        # self.workflow.add_edge("ToolCaller", "Supervisor")
        self.workflow.add_edge("ToolCaller", END)


        # Compile workflow
        self.memory = MemorySaver()
        self.graph = self.workflow.compile()


    def get_graph(self):
        """Return the compiled workflow graph"""
        return self.graph