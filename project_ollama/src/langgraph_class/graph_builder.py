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
from src.langgraph_class.states import AnalysisState

from src.nodes.preprocess_node import preprocess_node
from src.nodes.param_extractor_node import UserInputNode, ParamExtractorNode, MissingParamsHandlerNode
from src.nodes.workflow_node import *
from src.nodes.pandas_agent import PandasAgentNode
from src.nodes.retriever_node import RetrieverNode

from src.utils.config import TEST_WORKFLOW, TEST_PANDAS

def build_graph(llm, embedding):
    """
    Constructs and returns a compiled StateGraph for the analysis workflow.

    Args:
        llm: Language model for natural language processing tasks
        embedding: Embedding model for text vectorization

    Returns:
        A compiled StateGraph object representing the analysis workflow
    """
    graph = StateGraph(AnalysisState)

    # Define nodes for various stages of the workflow
    nodes = {
        # Data preproessing
        "PreprocessNode": preprocess_node,

        # User interaction and parameter extraction
        "UserInputNode": UserInputNode(),
        "ParamExtractorNode": ParamExtractorNode(llm),
        "MissingParamsHandlerNode": MissingParamsHandlerNode(),

        # Task selection
        "TaskDecisionNode": TaskDecisionNode(),
        
        # Task workflow
        "FindLargestNode": FindLargestNode(),
        "FindLargestWithinHaloNode": FindLargestWithinHaloNode(),
        "TrackObjectEvolutionNode": TrackObjectEvolutionNode(),

        # Post task nodes - making visualizations
        "CheckVisualNode": CheckVisualNode(),
        "VisualizeNode": VisualizeNode(),
        "MultiTimestepVisualizeNode": MultiTimestepVisualizeNode(),
        "UserVisualizePromptNode": UserVisualizePromptNode(),

        # Data exploration
        "UserDataExplorerPromptNode": UserDataExplorerPromptNode(),
        "RetrieverNode": RetrieverNode(embedding),
        "PandasAgentNode": PandasAgentNode(llm),
    }

    # Add nodes to the graph
    for name, node in nodes.items():
        graph.add_node(name, node)

    # Configure start node based on testing settings
    if TEST_PANDAS:
        # For testing pandas agent
        graph.add_edge(START, "RetrieverNode")
    elif TEST_WORKFLOW:
        # For testing workflow
        graph.add_edge(START, "TaskDecisionNode")
    else:
        # Normal flow
        graph.add_edge(START, "PreprocessNode")
        graph.add_edge("PreprocessNode", END)


    # Define the main flow of the graph
    graph.add_edge("PreprocessNode", "UserInputNode")
    graph.add_edge("UserInputNode", "ParamExtractorNode")
    graph.add_edge("ParamExtractorNode", "MissingParamsHandlerNode")  # go here if read

    # Handle missing parameters
    graph.add_conditional_edges(
        "MissingParamsHandlerNode",
        lambda state: state.get("next_node", "UserInputNode"),
        {
            "UserInputNode": "UserInputNode",       # loop back
            "TaskDecisionNode": "TaskDecisionNode"  # exit loop
        }
    )

    # Workflow nodes - task selection and execution
    graph.add_conditional_edges(
        "TaskDecisionNode", 
        lambda state: state.get("task_type"), 
        {   
            "find_largest_object": "FindLargestNode", 
            "find_largest_within_halo": "FindLargestWithinHaloNode",
            "track_object_evolution": "TrackObjectEvolutionNode"
        }
    )

    # Connect task nodes to visualization check
    graph.add_edge("FindLargestNode", "CheckVisualNode")
    graph.add_edge("FindLargestWithinHaloNode","CheckVisualNode") 
    graph.add_edge("TrackObjectEvolutionNode", "CheckVisualNode")

    # Handle visualization based on user preference
    graph.add_conditional_edges(
        "CheckVisualNode",
        lambda state: state.get("use_visual", False),
        {True: "VisualizeNode", False: "UserVisualizePromptNode"}
    )

    # Handle different visualizations
    graph.add_conditional_edges(
        "UserVisualizePromptNode",
        lambda state: (
            "MultiTimestepVisualizeNode" if (state.get("track_evolution", False) and state.get("start_visual", False)) else
            "VisualizeNode" if state.get("start_visual", False) else
            "default_route"
        ),
        {
            "MultiTimestepVisualizeNode": "MultiTimestepVisualizeNode", 
            "VisualizeNode": "VisualizeNode",
            "default_route":"UserDataExplorerPromptNode"
        }
    )

    # Connect visualization nodes to data explorer prompt
    graph.add_edge("VisualizeNode", "UserDataExplorerPromptNode")
    graph.add_edge("MultiTimestepVisualizeNode", "UserDataExplorerPromptNode")

    # Handle data exploration
    graph.add_conditional_edges(
        "UserDataExplorerPromptNode",
        lambda state: state.get("start_explorer"),
        {True: "RetrieverNode", False: END}
    )

    graph.add_edge("RetrieverNode", "PandasAgentNode")

    # Connect pandas agent to the end of workflow
    # graph.add_edge("PandasAgentNode", END)

    return graph.compile()