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

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from src.core.state import State
from src.nodes import (
    dataloader_node, 
    human_feedback_node, 
    verifier_node, 
    planner_node, 
    supervisor_node,
    retriever_node
)

from src.tools import (
    dataload_tools, 
    routing_tools
)

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
        self.config = {"configurable": {"thread_id": "1"}}
        # self.members = ["toolcalling_agent"]
        self.tools = self.create_tools()
        self.agents = self.create_agents()
        self.setup_workflow()


    def create_tools(self):
        tools = {}
        tools["dataloader_tools"] = [dataload_tools.load_file_index]
        tools["routing_tools"] = [routing_tools.redirect]

        return tools

    def create_agents(self):
        """Create all the agents used by the workflow"""
        llm = self.language_models["llm"]
        power_llm = self.language_models["power_llm"]
        json_llm = self.language_models["json_llm"]
        embed_llm = self.language_models["embed_llm"]

        # Dictionary of agents
        agents = {}
        # Create agents using different LLMs depending on their function
        agents["DataLoader"] = dataloader_node.Node(llm, self.tools["dataloader_tools"])
        agents["HumanFeedback"] = human_feedback_node.Node()
        agents["Planner"] = planner_node.Node(power_llm)
        agents["Verifier"] = verifier_node.Node(llm , self.tools["routing_tools"])
        agents["Supervisor"] = supervisor_node.Node(llm, self.tools["routing_tools"])
        agents["Retriever"] = retriever_node.Node(embed_llm)

        return agents
    

    def setup_workflow(self):
        self.workflow = StateGraph(State)
        """Setup langgraph graph as workflow"""

        # Add nodes
        self.workflow.add_node("DataLoader", self.agents["DataLoader"])
        self.workflow.add_node("HumanFeedback", self.agents["HumanFeedback"])
        self.workflow.add_node("Planner", self.agents["Planner"])
        self.workflow.add_node("Verifier", self.agents["Verifier"])
        self.workflow.add_node("Supervisor", self.agents["Supervisor"])

        self.workflow.add_node("DataLoaderTool", ToolNode(self.tools["dataloader_tools"]))
        self.workflow.add_node("RoutingTool", ToolNode(self.tools["routing_tools"]))


        self.workflow.add_edge(START, "DataLoader")
        self.workflow.add_conditional_edges(
            "DataLoader",
            lambda x: x['next'], 
            {
                "HumanFeedback": "HumanFeedback", 
                "DataLoaderTool": "DataLoaderTool"
            }
        )

        self.workflow.add_conditional_edges(
            "DataLoaderTool",
            lambda x: x["messages"][-1].status,
            {
                "error": "HumanFeedback",
                "success": "Planner"
            }
        )

        self.workflow.add_edge("Planner", "Verifier")
    
        self.workflow.add_conditional_edges(
            "Verifier",
            lambda x: x["next"],
            {
                "HumanFeedback": "HumanFeedback",
                "RoutingTool": "RoutingTool"
            }
        )
        self.workflow.add_edge("Supervisor", "RoutingTool")


        self.workflow.add_conditional_edges(
            "RoutingTool",
            lambda x: x["next"],
            {
                "Planner": "Planner",
                "Supervisor": "Supervisor",
                "SQLProgrammer": END,
                "PythonProgrammer": END,
                "Visualization": END
            }
        )

        self.workflow.add_conditional_edges("HumanFeedback", 
            lambda x: x["next"],
            {
                "DataLoader": "DataLoader",
                "Planner": "Planner",
                "Verifier": "Verifier"
            }
        )


        # Compile workflow
        self.memory = InMemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)


    def get_graph(self):
        """Return the compiled workflow graph"""
        return self.graph