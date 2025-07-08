from langchain_core.messages import convert_to_messages
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, 'content'):
        # likely HumanMessage, AIMessage, etc.
        return {"type": obj.__class__.__name__, "content": obj.content}
    else:
        return str(obj)  # fallback
    

def reconstruct_message(obj):
    if isinstance(obj, list):
        return [reconstruct_message(item) for item in obj]
    if isinstance(obj, dict) and "type" in obj and "content" in obj:
        msg_type = obj["type"]
        content = obj["content"]
        if msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "AIMessage":
            return AIMessage(content=content)
        elif msg_type == "SystemMessage":
            return SystemMessage(content=content)
        else:
            # Unknown type — fallback to dict or raise error
            return obj
    return obj