"""
Utilities for pretty-printing LangChain messages and updates with structured indentation.

These functions help log messages in a readable, formatted style with optional indentation for nested subgraph updates.

Usage example:
    from your_module import pretty_print_messages

    # Log updates normally
    pretty_print_messages(update_data)

    # Log only the last message from each node
    pretty_print_messages(update_data, last_message=True)
"""

import logging
from langchain_core.messages import convert_to_messages

logger = logging.getLogger(__name__)

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        logger.info(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    
    logger.info(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        logger.info(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        logger.info(f"{update_label}\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        