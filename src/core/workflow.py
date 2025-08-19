"""
Module: graph_builder.py

Purpose:
    Constructs a LangGraph-based workflow graph for data analysis using multiple
    LangChain agents (nodes) with capabilities like planning, code generation,
    feedback handling, and visualization.

Graph Structure:
    - EntryPoint: Initializes the workflow.
    - Planner: Determines what task to do.
    - Supervisor: Controls high-level routing.
    - Task Nodes: Specialized nodes for data loading, coding, visualization, etc.
    - QA & Feedback Nodes: Perform quality checks or receive human input.
    - Documentation & Summary: Record outputs and generate reports.

Usage:
    >>> manager = WorkflowManager(language_models, working_directory)
    >>> graph = manager.get_graph()

How to Add a New Node to the Workflow:
    1. Define your new agent class (in `src/nodes/your_node.py`) and ensure it inherits from `NodeBase`.
    2. Add your node to `create_agents()` by initializing it.
    3. Register the node with `add_node()` in `setup_workflow()`.
    4. Add relevant edges using:
        - `add_edge(from_node, to_node)`
        - or `add_conditional_edges(from_node, lambda state: ..., {"condition": "next_node"})`
    5. Optional: Register tools in `create_tools()` and use `ToolNode()` if your node is tool-driven.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from src.core.state import State
from src.nodes.node_base import NodeBase

# Node modules
from src.nodes import (
    preprocess_node,
    human_feedback_node, planner_node, supervisor_node,
    dataloader_node, retriever_node,
    sql_node, python_node, visualization_node,
    QA_node, documentation_node, summary_node,
)

from src.tools import (
    dataload_tools, 
    routing_tools,
    python_tools,
    custom_tools,
)




class WorkflowManager:
    def __init__(self, language_models, working_directory):
        """
        Initialize the workflow manager.

        Args:
            language_models (dict): Dictionary with LLMs: llm, power_llm, json_llm, embed.
            working_directory (str): Path where files/results are stored.
        """
        self.language_models = language_models
        self.working_directory = working_directory
        self.workflow = None
        self.memory = None
        self.graph = None
        self.config = {"configurable": {"thread_id": "1"}}
        self.tools = self.create_tools()
        self.agents = self.create_agents()
        self.setup_workflow()


    def create_tools(self):
        """
        Register tools used by tool-driven nodes.
        Tools can be reused across multiple nodes or attached to ToolNode instances.
        """
        tools = {}
        tools["dataloader_tools"] = [dataload_tools.load_file_index]
        tools["db_writer"] = [dataload_tools.load_to_db]
        tools["routing_tools"] = [routing_tools.redirect]
        tools["python_tools"] = [python_tools.GenerateCode, custom_tools.track_halo_evolution]
        tools["dataframe_tools"] = [python_tools.LoadDataframes]
        tools["visual_tools"] = [python_tools.GenerateVisualization, custom_tools.generate_pvd_file]

        return tools

    def create_agents(self):
        """
        Instantiate all agent nodes (LangGraph nodes).
        These should inherit from NodeBase or implement a .run(state) method.
        """
        server = self.language_models["server"]
        llm = self.language_models["llm"]
        power_llm = self.language_models["power_llm"]
        json_llm = self.language_models["json_llm"]
        embed_llm = self.language_models["embed_llm"]

        agents = {}

        agents["Preprocess"] = preprocess_node.Node(llm)

        agents["HumanFeedback"] = human_feedback_node.Node()
        agents["QA"] = QA_node.Node(power_llm)
        agents["Summary"] = summary_node.Node(llm)
        agents["Documentation"] = documentation_node.Node(llm)

        agents["Planner"] = planner_node.Node(power_llm)
        agents["Supervisor"] = supervisor_node.Node(llm, self.tools["routing_tools"])

        agents["DataLoader"] = dataloader_node.Node(llm, self.tools["dataloader_tools"], self.tools["db_writer"])
        agents["Retriever"] = retriever_node.Node(embed_llm, server)

        agents["SQLProgrammer"] = sql_node.Node(llm)
        agents["PythonProgrammer"] = python_node.Node(llm, self.tools["python_tools"], self.tools["dataframe_tools"])
        agents["Visualization"] = visualization_node.Node(llm, self.tools["visual_tools"])

        return agents

    def setup_workflow(self):
        """
        Set up the workflow as a LangGraph StateGraph using defined agents and tools.
        This defines the flow, branching logic, and error handling strategy.
        """
        self.workflow = StateGraph(State)

        # Entry point node for initializing state
        class EntryPoint(NodeBase):
            def __init__(self):
                super().__init__("EntryPoint")

            def run(self, state):
                return state


        # === Register LangGraph Nodes ===
        self.workflow.add_node("EntryPoint", EntryPoint(), destinations=["Planner"])

        # Registers all agents
        for name in self.agents:
            self.workflow.add_node(name, self.agents[name])

        # ToolNodes allow routing or tool-execution logic
        self.workflow.add_node("DataLoaderTool", ToolNode(self.tools["dataloader_tools"]))
        self.workflow.add_node("RoutingTool", ToolNode(self.tools["routing_tools"]))
        self.workflow.add_node("DBWriter", ToolNode(self.tools["db_writer"]))
        self.workflow.add_node("PythonTool", ToolNode(self.tools["python_tools"]))
        self.workflow.add_node("VisualTool", ToolNode(self.tools["visual_tools"]))


        # === Edges: Define Graph Flow ===

        # Entry logic — from START to initial decision
        self.workflow.add_edge(START, "EntryPoint")
        self.workflow.add_conditional_edges("EntryPoint", lambda x: x["next"], {}) # automatically goes to ["next"]

        self.workflow.add_edge("Preprocess", "Planner")

        # Planner: Chooses between Documentation and HumanFeedback - Documentation means its complete
        self.workflow.add_conditional_edges("Planner", lambda x: x["next"], {
            "Documentation": "Documentation",
            "HumanFeedback": "HumanFeedback"
        })

        # Supervisor: routes to RoutingTool or END
        self.workflow.add_conditional_edges("Supervisor", lambda x: x['next'], {
            "RoutingTool": "RoutingTool",
            "END": END
        })
        
        
        # DataLoader: decision flow
        self.workflow.add_conditional_edges("DataLoader", lambda x: x['next'], {
            "HumanFeedback": "HumanFeedback", 
            "DataLoaderTool": "DataLoaderTool",
            "DBWriter": "DBWriter",
            "Retriever": "Retriever",
            "Documentation": "Documentation"
        })

        # Retriever: retrieves context and is always followed by DataLoader
        self.workflow.add_conditional_edges("Retriever", lambda x: x["next"], {   
            "DataLoader": "DataLoader",
        })

        # Handle success/failure from tool nodes (DataLoaderTool and DBWriter)
        self.workflow.add_conditional_edges("DataLoaderTool", lambda x: x["messages"][-1].status, {
            "error": "HumanFeedback",
            "success": "DataLoader"
        })
        self.workflow.add_conditional_edges("DBWriter", lambda x: x["messages"][-1].status, {
            "error": "HumanFeedback",
            "success": "DataLoader"
        })


        # Python: code generation → Tool → QA
        self.workflow.add_conditional_edges("PythonProgrammer", lambda x: x['next'], {
            "PythonTool": "PythonTool", 
            "QA": "QA"
        })
        self.workflow.add_conditional_edges("PythonTool", lambda x: x["messages"][-1].status, {
            "error": "HumanFeedback",
            "success": "QA",
        })


        # Visualization: logic flow
        self.workflow.add_conditional_edges("Visualization", lambda x: x['next'], {
            "VisualTool": "VisualTool", 
            "QA": "QA"
        })
        self.workflow.add_conditional_edges("VisualTool", lambda x: x["messages"][-1].status, {
            "error": "HumanFeedback",
            "success": "QA",
        })

        # SQL output always routes to QA - no tool call
        self.workflow.add_edge("SQLProgrammer", "QA")

        # Route from RoutingTool to appropriate node - decided by supervisor
        self.workflow.add_conditional_edges("RoutingTool", lambda x: x["next"], {
            "Planner": "Planner",
            "DataLoader": "DataLoader",
            "Documentation": "Documentation",
            "SQLProgrammer": "SQLProgrammer",
            "PythonProgrammer": "PythonProgrammer",
            "Visualization": "Visualization",
            "Summary": "Summary",
            "HumanFeedback": "HumanFeedback",
            "END": END
        })

        # HumanFeedback can go anywhere; automatically goes to ["next"]
        self.workflow.add_conditional_edges("HumanFeedback", lambda x: x["next"],{})

        # Summary is followed by documentation
        self.workflow.add_edge("Summary", "Documentation")

        # Documentation is followed by supervisor
        self.workflow.add_edge("Documentation", "Supervisor")

        # QA routes based on retry logic or success path
        self.workflow.add_conditional_edges("QA", lambda x: x["next"], {   
            "Documentation": "Documentation",
            "SQLProgrammer": "SQLProgrammer",
            "DataLoader": "DataLoader",
            "PythonProgrammer": "PythonProgrammer",
            "Visualization": "Visualization"
        })

        # Compile workflow
        self.memory = InMemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)


    def get_graph(self):
        """Return the compiled workflow graph"""
        return self.graph