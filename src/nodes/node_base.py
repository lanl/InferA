"""
Module: node_base.py

Purpose:
    This module defines a base class `NodeBase` that serves as a foundational interface
    for all processing nodes in a pipeline or modular system. It handles standardized
    logging and provides a structure that subclasses can extend to implement specific
    behaviors.

Classes:
    - NodeBase: Abstract base class for nodes. Provides logging and a callable interface.

Usage:
    Inherit from `NodeBase` and implement the `run(self, state)` method in your subclass.

Example:
    class MyCustomNode(NodeBase):
        def run(self, state):
            # Implement custom processing logic here
            return updated_state
"""

import logging

logger = logging.getLogger(__name__)

class NodeBase:
    """
    Abstract base class for creating nodes with standardized logging.

    Attributes:
        name (str): Name of the node for logging identification.
        logger (logging.Logger): Logger instance used for this node.

    Methods:
        __call__(state): Handles pre- and post-processing logging, then delegates to run().
        run(state): Abstract method to be implemented by subclasses.
    """
    def __init__(self, name: str):
        """
        Initializes the NodeBase instance.

        Args:
            name (str): Human-readable name for the node, used in logs.
        """
        self.name = name
        self.logger = logger

    def __call__(self, state):
        """
        Callable interface for the node. Logs input and output states,
        and invokes the subclass's `run` method.

        Args:
            state: Arbitrary input data/state for the node to process.

        Returns:
            The result of the `run` method.
        """
        self.logger.info(f"\033[1;34m[{self.name.upper()}] Starting node...\033[0m")
        self.logger.trace(f"[{self.name.upper()}] Input state: {state}")

        result = self.run(state)

        self.logger.info(f"\033[1;34m[{self.name.upper()}] Finished node.\033[0m")
        self.logger.trace(f"[{self.name.upper()}] Output state: {result}")

        return result

    def run(self, state):
        """
        Method to be implemented by subclasses with the node's core logic.

        Args:
            state: The input state/data for the node.

        Returns:
            Modified state/data after processing.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement run()")
    


# === Customization Instructions ===
# To use this base class, create a subclass that implements the `run()` method:
#
# class MyNode(NodeBase):
#     def run(self, state):
#         # Modify state here
#         return state
#
# Replace `MyNode` and its logic to suit your specific node behavior.
