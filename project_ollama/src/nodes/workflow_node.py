# workflow_node.py

"""
This module defines a set of workflow nodes that form part of a data processing and analysis pipeline.

Each node inherits from NodeBase and implements a `run(state)` method to perform specific tasks.

These nodes operate by reading from and updating a shared state dictionary in states.py, enabling flexible routing and chaining of workflow steps.
"""

import os
from tqdm import tqdm
from src.langgraph_class.node_base import NodeBase
from src.workflows import *
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

class TaskDecisionNode(NodeBase):
    """
    Node responsible for deciding and routing the next workflow task to execute based on the current state.

    This node inspects the 'task_type' in the state and updates the state to route
    to the corresponding workflow node. It also clears the 'message' to prepare for
    the next step.
    """
    def __init__(self):
        super().__init__("TaskDecisionNode")

    def run(self, state):
        """
        Execute the decision logic to determine the next workflow node.

        Args:
            state (dict): The current state dictionary containing at least 'task_type' and 'parameters'.
        """
        logger.info(f"Executing task: {state["task_type"]} with params: {state["parameters"]}")

        # After execution, clear to prepare for next query
        state["message"] = []

        # Set next node in workflow based on task_type
        state["next_node"] = state.get("task_type")

        logger.info(f"Routing to workflow: {state["next_node"]}")
        return state
    

class FindLargestNode(NodeBase):
    """
    Node that finds the largest objects of a given type within the dataset.

    It leverages the 'FindLargestObject' workflow component to perform the computation.
    """
    def __init__(self):
        super().__init__("FindLargestNode")

    def run(self, state):
        """
        Runs the largest_object_finder workflow.

        Args:
            state (dict): The current state containing 'parameters' with keys:
                          'object_type', 'timestep', and 'n' (number of objects to find).

        Returns:
            dict: Updated state with 'result' containing the DataFrame of largest objects.
        """
        parameters = state.get("parameters")
        object_type = parameters["object_type"]
        timestep = parameters["timestep"]
        n = parameters["n"]

        # Initialize finder class with base dir
        largest_finder = find_largest_object.FindLargestObject(state.get("base_dir"))

        # Run finder and get results as dataframe
        df = largest_finder.run(object_type, timestep, n)

        state["result"] = df
        return state
    

class VisualizeNode(NodeBase):
    """
    Node that visualizes the results of previous computations.

    Uses the 'VisualizeObject' workflow component to generate visualizations from dataframes and save output files.
    """
    def __init__(self):
        super().__init__("VisualizeNode")

    def run(self, state):
        """
        Runs the visualization process based on the results stored in the state.

        Args:
            state (dict): The current state containing:
                          - 'parameters' with 'object_type', 'timestep', 'n'
                          - 'result': DataFrame with the data to visualize
                          - 'base_dir' and 'full_dir' paths for file operations

        Returns:
            dict: Updated state with 'visual_output' holding the output filename and path.
        """
        parameters = state.get("parameters")
        object_type = parameters["object_type"]
        timestep = parameters["timestep"]

        # Get dataframe result to visualize
        df = state.get("result")

        # Initialize and run visualization class - returns output file path
        visualize = visualize_object.VisualizeObject(state.get("base_dir"), state.get("full_dir"))
        output_file = visualize.run(df, object_type, timestep, "plot")

        print(f"✅ Visualization written to {output_file}. You can download file to view.")


        state["visual_output"] = output_file
        return state
    

class MultiTimestepVisualizeNode(NodeBase):
    def __init__(self):
        super().__init__("VisualizeNode")

    def run(self, state):
        parameters = state.get("parameters")
        object_type = parameters["object_type"]

        # Get dataframe result to visualize
        df = state.get("evolution_result")

        output_dir = "output/multitimestep"
        os.makedirs(output_dir, exist_ok=True)

        visual_output = []
        visualize = visualize_object.VisualizeObject(state.get("base_dir"), state.get("full_dir"))

        for idx, row in tqdm(enumerate(df), total=len(df), desc = "Outputting visualizations: "):
            # timestep = row["timestep"]
            # Wrap single row as DataFrame

            try:
                timestep = row["timestep"].iloc[0]
                # Initialize and run visualization class - returns output file path
                output_file = visualize.run(row, object_type, timestep, f"multitimestep/{str(timestep)}")
                visual_output.append(output_file)

            except Exception as e:
                logger.info(f"[ERROR] Error on timestep {timestep}: {e}")

        # Generate .pvd file pointing to all visualization files
        pvd_path = os.path.join(output_dir, "series.pvd")
        visualize.generate_pvd_file(visual_output, pvd_path)

        # Append .pvd file to list of outputs for zipping
        visual_output.append(pvd_path)

        # Create zip archive of all output files
        zip_path = visualize.create_zip_archive(visual_output, output_dir="output/", zip_name="multitimestep_visualizations.zip")
        
        logger.info(f"[MultiTimestepVisualizeNode] Wrote to {zip_path}")
        print(f"✅ Created zip archive of visualizations at: {zip_path}, You can download and view in Paraview.")

        return state



class FindLargestWithinHaloNode(NodeBase):
    """
    Node that finds the largest objects of a given type within a specific halo.

    Uses the 'FindLargestWithinHalo' workflow component for filtering and selection.
    """
    def __init__(self):
        super().__init__("FindLargestWithinHaloNode")

    def run(self, state):
        """
        Runs the largest-within-halo finder workflow.

        Args:
            state (dict): Current state containing 'parameters' with keys:
                          'object_type', 'timestep', 'halo_id', and 'n'.

        Returns:
            dict: Updated state with 'result' DataFrame of largest objects within the specified halo.
        """
        parameters = state.get("parameters")
        object_type = parameters["object_type"]
        timestep = parameters["timestep"]
        halo_id = parameters["halo_id"]
        n = parameters["n"]

        # Initialize and run finder with base dir - returns a dataframe
        largest_within_halo_finder = find_largest_within_halo.FindLargestWithinHalo(state.get("base_dir"))
        df = largest_within_halo_finder.run(object_type, timestep, halo_id, n)

        state["evolution_result"] = df
        return state
    

class TrackObjectEvolutionNode(NodeBase):
    def __init__(self):
        super().__init__("FindLargestWithinHaloNode")

    def run(self, state):
        parameters = state.get("parameters")
        object_type = parameters["object_type"]
        object_id = parameters["object_id"]
        timestep = parameters["timestep"]

        # Initialize and run finder with base dir - returns a dataframe
        object_tracker = track_object_evolution.TrackObjectEvolution(state.get("base_dir"))
        list_df, compiled_df = object_tracker.run(object_type, object_id, timestep)

        state["result"] = compiled_df
        state["evolution_result"] = list_df
        state["track_evolution"] = True
        return state
    

class CheckVisualNode(NodeBase):
    """
    Node that prompts the user to decide whether to visualize the results.
    """
    def __init__(self):
        super().__init__("CheckVisualNode")

    def run(self, state):
        use_visual = state.get("use_visual", False)
        logger.info(f"[CheckVisualizationNode] use_visual flag is set to: {use_visual}")
        return state


class UserVisualizePromptNode(NodeBase):
    """
    Node that prompts the user to decide whether to visualize the results.
    """
    def __init__(self):
        super().__init__("UserVisualizePromptNode")

    def run(self, state):
        """
        Prompt user for visualization consent until valid input ('y' or 'n') is received.

        Args:
            state (dict): The current state to be updated with 'start_visual' boolean.

        Returns:
            dict: Updated state with 'start_visual' key set to True or False.
        """
        while True:
            user_input = input(f"\nDo you want to visualize the results? (y/n): \n").strip().lower()
            if user_input in ('y', 'n'):
                state["start_visual"] = True if user_input == 'y' else False
                logger.info(f"[UserVisualizePromptNode] User chose to {'visualize' if user_input == 'y' else 'skip visualization.'}\n")
                return state
            print("❌ Invalid input. Please enter 'y' or 'n'.")


class UserDataExplorerPromptNode(NodeBase):
    """
    Node that prompts the user to decide whether to explore the results using an agent.
    """
    def __init__(self):
        super().__init__("UserDataExplorerPromptNode")

    def run(self, state):
        """
        Prompt user for data exploration consent until valid input ('y' or 'n') is received.

        Args:
            state (dict): The current state to be updated with 'start_explorer' boolean.

        Returns:
            dict: Updated state with 'start_explorer' key set to True or False.
        """
        while True:
            user_input = input(f"\nDo you want to explore the results with an agent? (y/n): \n").strip().lower()
            if user_input in ('y', 'n'):
                state["start_explorer"] = True if user_input == 'y' else False
                logger.info(f"[UserDataExplorerPromptNode] User chose to {'continue exploring' if user_input == 'y' else 'skip exploring'}\n")
                return state
            print("❌ Invalid input. Please enter 'y' or 'n'.")


# class PandasAgentNode(NodeBase):
#     def __init__(self):
#         super().__init__("PandasAgentNode")

#     def run(self, state):
#         logger.info("PandasAI agent placeholder...")