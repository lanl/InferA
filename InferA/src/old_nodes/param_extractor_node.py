import re
import logging
from typing import Tuple, List, Dict, Any

from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict

from src.langgraph_class.node_base import NodeBase
from src.workflows.registry import required_fields_by_task

logger = logging.getLogger(__name__)

class UserInputNode(NodeBase):
    def __init__(self):
        super().__init__("UserInputNode")

    def run(self, state):
        task = state.get("task_type", None)
        missing = state.get("missing_parameters", [])
        params = state.get("parameters", [])

        if not task:
            prompt = "\n\033[1m\033[31mYou - Enter your data query (or 'exit' to quit): \033[0m\n"
            user_input = input(prompt)
        else:
            prompt = f"\nIt looks like you would like to use the workflow [{task}]. However, required info is missing.\n\033[31m\033[1mPlease provide {', '.join(missing)}. You may list the parameters in the form <parameter> = <input> \n\033[0m"
            update_input = input(prompt)
            user_input = f"SUPPLEMENTAL INFORMATION: {update_input}"

        if user_input.lower() in {"exit", "quit"}:
            state["exit"] = True
            return state
        # Append new input to message history
        state.setdefault("message", [])
        state["message"].append(user_input)
        state["user_input"] = user_input

        return state


class ParamExtractorNode(NodeBase):
    """
    Node class that takes a user query string and outputs the task
    and extracted parameters along with any missing fields.
    """
    def __init__(self, llm):
        super().__init__("ParamExtractorNode")
        self.llm = llm
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("message", [])
        if not messages:
            raise ValueError("No message history found in state.")

        full_input = "\n".join(messages)

        # Run extraction
        params, missing, task, use_visual = extract_params(self.llm, full_input)
        logger.info(f"[{self.name}] Current parameter state: {params}")


        prev_task = state.get("task_type", None)
        if prev_task and task != prev_task:
            logger.info(f"[{self.name}] Task changed: [{prev_task}] to [{task}]. Resetting message history.")
            print(f"🟢 Based on your most recent query, the task has been changed from [{prev_task}] to [{task}].")

            state["message"] = [messages[-1]]  # Reset message to current input only

        # Update state fields
        state["task_type"] = task
        state["parameters"] = params
        state["missing_parameters"] = missing
        state["use_visual"] = use_visual

        return state


class MissingParamsHandlerNode(NodeBase):
    def __init__(self):
        super().__init__("MissingParamsHandlerNode")

    def run(self, state):
        task = state.get("task_type", "unknown task")
        missing = state.get("missing_parameters", [])

        if not missing and task:
            state["next_node"] = "TaskDecisionNode"
            logger.info(f"[{self.name}] All parameters gathered. Routing to TaskDecisionNode.")
            print(f"✅ All parameters have been provided. Executing workflow...")
            return state

        state["next_node"] = "UserInputNode"
        logger.info(f"[{self.name}] Missing params detected. Routing back to UserInputNode for supplemental information.")

        return state
    

class FileParam(BaseModel):
    model_config = ConfigDict(
        extra="allow"
    )



def generate_task(llm, query:str) -> str:
    prompt = PromptTemplate(
        template =
        "You're a cosmology data scientist that classifies the user's request into a known task category.\n"
        "Query: {query}\n"
        "Available tasks: [find_largest_object, find_largest_within_halo (only do this if user specifically asks to filter for objects within a halo), track_object_evolution (do this if you need to track an object through all timesteps)]\n"
        "Output the classification label in curly braces. Only choose one task."
        "Task: ",
        input_variables=["query"],
    )
    return (prompt | llm).invoke({"query":query})


def generate_task_parameters(llm, query:str, task:str, params: list) -> Dict:
    # llm_json = llm.bind(format="json")
    llm_json = llm
    parser = JsonOutputParser(pydantic_object=FileParam)
    
    prompt = PromptTemplate(
        template =
        "You are a cosmology data scientist that is extracting parameters for a specific analysis task on HACC data.\n"
        "Task: {task}\n"
        "Parameters to extract: {params}\n"
        "User query:{query}\n"
        "{format_instructions}\n"
        "Extract all the required parameters for this task. Respond using JSON."
        "Return null or empty if a parameter is missing. e.g. object_type:''"
        "Ex. Query: 'Find the largest object' -> Required parameters=object_type:None, timestep:None.",
        input_variables=["query", "task", "params"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return (prompt|llm_json|parser).invoke({"query": query, "task": task, "params": params})


def get_task_params(task: str) -> Tuple[str, List[str]]:
    match = re.search(r"\{(.*?)\}", task)
    if match:
        task = match.group(1)
        params = required_fields_by_task[task]
        return task, params
    else:
        raise ValueError("Error finding task.")


def extract_params(llm, query: str) -> Tuple[dict, list[str]]:
    # Step 1: Generate task using LLM based on user query
    task = generate_task(llm, query)

    # Step 2: Parse required parameters to complete task workflow
    task, required_params = get_task_params(task.content)

    # Step 3: Fill in parameters using LLM based on user query
    params = generate_task_parameters(llm, query, task, required_params)
    print(params)

    # Step 4: Check for missing fields - if fields missing, return a string to user to add additional information. If no fields missing, proceed to workflow.
    missing = [param for param, value in params.items() if value is None]

    use_visual = params["use_visual"]

    return params, missing, task, use_visual
