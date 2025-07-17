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
from src.nodes.node_base import NodeBase

from src.nodes import (
    dataloader_node, 
    human_feedback_node, 
    verifier_node, 
    planner_node, 
    supervisor_node,
    retriever_node,
    sql_node,
    QA_node,
    python_node,
    summary_node,
    visualization_node
)

from src.tools import (
    dataload_tools, 
    routing_tools,
    python_tools,
    custom_tools,
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
        self.tools = self.create_tools()
        self.agents = self.create_agents()
        self.setup_workflow()


    def create_tools(self):
        tools = {}
        tools["dataloader_tools"] = [dataload_tools.load_file_index]
        tools["db_writer"] = [dataload_tools.load_to_db]
        tools["routing_tools"] = [routing_tools.redirect]
        tools["python_tools"] = [python_tools.GenerateCode, custom_tools.track_halo_evolution]
        tools["visual_tools"] = [python_tools.GenerateVisualization, custom_tools.generate_pvd_file]

        return tools

    def create_agents(self):
        """Create all the agents used by the workflow"""
        server = self.language_models["server"]
        llm = self.language_models["llm"]
        power_llm = self.language_models["power_llm"]
        json_llm = self.language_models["json_llm"]
        embed_llm = self.language_models["embed_llm"]

        # Dictionary of agents
        agents = {}
        # Create agents using different LLMs depending on their function
        agents["HumanFeedback"] = human_feedback_node.Node()
        agents["QA"] = QA_node.Node(power_llm)
        agents["Summary"] = summary_node.Node(llm)

        agents["Planner"] = planner_node.Node(power_llm)
        agents["Verifier"] = verifier_node.Node(llm , self.tools["routing_tools"])
        agents["Supervisor"] = supervisor_node.Node(llm, self.tools["routing_tools"])

        agents["DataLoader"] = dataloader_node.Node(llm, self.tools["dataloader_tools"], self.tools["db_writer"])
        agents["Retriever"] = retriever_node.Node(embed_llm, server)
        agents["SQLProgrammer"] = sql_node.Node(llm)
        agents["PythonProgrammer"] = python_node.Node(llm, self.tools["python_tools"])
        agents["Visualization"] = visualization_node.Node(llm, self.tools["visual_tools"])

        return agents

    def setup_workflow(self):
        self.workflow = StateGraph(State)
        """Setup langgraph graph as workflow"""

        class EntryPoint(NodeBase):
            def __init__(self):
                super().__init__("EntryPoint")

            def run(self, state):
                return state

        # Add nodes
        self.workflow.add_node("EntryPoint", EntryPoint(), destinations=["Planner"])
        self.workflow.add_node("HumanFeedback", self.agents["HumanFeedback"])
        self.workflow.add_node("QA", self.agents["QA"])
        self.workflow.add_node("Summary", self.agents["Summary"])

        self.workflow.add_node("Planner", self.agents["Planner"])
        # self.workflow.add_node("Verifier", self.agents["Verifier"])
        self.workflow.add_node("Supervisor", self.agents["Supervisor"])

        self.workflow.add_node("Retriever", self.agents["Retriever"])
        self.workflow.add_node("DataLoader", self.agents["DataLoader"])

        self.workflow.add_node("SQLProgrammer", self.agents["SQLProgrammer"])
        self.workflow.add_node("PythonProgrammer", self.agents["PythonProgrammer"])
        self.workflow.add_node("Visualization", self.agents["Visualization"])

        self.workflow.add_node("DataLoaderTool", ToolNode(self.tools["dataloader_tools"]))
        self.workflow.add_node("RoutingTool", ToolNode(self.tools["routing_tools"]))
        self.workflow.add_node("DBWriter", ToolNode(self.tools["db_writer"]))
        self.workflow.add_node("PythonTool", ToolNode(self.tools["python_tools"]))
        self.workflow.add_node("VisualTool", ToolNode(self.tools["visual_tools"]))


        # START is stateless. Use entrypoint to initialize start for initial routing: from previous state or stateless
        self.workflow.add_edge(START, "EntryPoint")

        self.workflow.add_conditional_edges(
            "EntryPoint",
            lambda x: x["next"],
            {}
        )

        self.workflow.add_conditional_edges(
            "Planner",
            lambda x: x["next"],
            {
                "Supervisor": "Supervisor",
                "HumanFeedback": "HumanFeedback"
            }
        )

        self.workflow.add_conditional_edges(
            "Supervisor",
            lambda x: x['next'], 
            {
                "RoutingTool": "RoutingTool",
                "END": END
            }
        )
        
        self.workflow.add_conditional_edges(
            "DataLoader",
            lambda x: x['next'], 
            {
                "HumanFeedback": "HumanFeedback", 
                "DataLoaderTool": "DataLoaderTool",
                "DBWriter": "DBWriter",
                "Retriever": "Retriever",
                "Supervisor": "Supervisor"
            }
        )

        self.workflow.add_conditional_edges(
            "DataLoaderTool",
            lambda x: x["messages"][-1].status,
            {
                "error": "HumanFeedback",
                "success": "DataLoader"
            }
        )

        self.workflow.add_conditional_edges(
            "DBWriter",
            lambda x: x["messages"][-1].status,
            {
                "error": "HumanFeedback",
                "success": "DataLoader"
            }
        )

        self.workflow.add_conditional_edges(
            "PythonProgrammer",
            lambda x: x['next'], 
            {
                "PythonTool": "PythonTool", 
                "QA": "QA"
            }
        )

        self.workflow.add_conditional_edges(
            "PythonTool",
            lambda x: x["messages"][-1].status,
            {
                "error": "HumanFeedback",
                "success": "QA",
            }
        )

        self.workflow.add_conditional_edges(
            "Visualization",
            lambda x: x['next'], 
            {
                "VisualTool": "VisualTool", 
                "QA": "QA"
            }
        )

        self.workflow.add_conditional_edges(
            "VisualTool",
            lambda x: x["messages"][-1].status,
            {
                "error": "HumanFeedback",
                "success": "QA",
            }
        )

        self.workflow.add_edge("SQLProgrammer", "QA")

        

        self.workflow.add_conditional_edges(
            "RoutingTool",
            lambda x: x["next"],
            {
                "Planner": "Planner",
                "DataLoader": "DataLoader",
                "Supervisor": "Supervisor",
                "SQLProgrammer": "SQLProgrammer",
                "PythonProgrammer": "PythonProgrammer",
                "Visualization": "Visualization",
                "Summary": "Summary",
                "HumanFeedback": "HumanFeedback",
                "END": END
            }
        )

        self.workflow.add_conditional_edges("HumanFeedback", 
            lambda x: x["next"],
            {}
        )

        # self.workflow.add_conditional_edges("HumanFeedback", 
        #     lambda x: x["next"],
        #     {
        #         "Supervisor": "Supervisor",
        #         "DataLoader": "DataLoader",
        #         "Planner": "Planner",
        #         "Verifier": "Verifier",
        #         "SQLProgrammer": "SQLProgrammer",
        #         "END": END
        #     }
        # )

        self.workflow.add_edge("Summary", "Supervisor")

        self.workflow.add_conditional_edges(
            "QA",
            lambda x: x["next"],
            {   
                "Supervisor": "Supervisor",
                "SQLProgrammer": "SQLProgrammer",
                "DataLoader": "DataLoader",
                "PythonProgrammer": "PythonProgrammer",
                "Visualization": "Visualization"
            }
        )

        self.workflow.add_conditional_edges(
            "Retriever",
            lambda x: x["next"],
            {   
                "DataLoader": "DataLoader",
            }
        )


        # Compile workflow
        self.memory = InMemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)


    def get_graph(self):
        """Return the compiled workflow graph"""
        return self.graph