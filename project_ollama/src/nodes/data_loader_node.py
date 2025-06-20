import os
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict

from src.langgraph_class.node_base import NodeBase
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

class DataLoaderNode(NodeBase):
    """
    Node responsible for deciding and routing the next workflow task to execute based on the current state.

    This node inspects the 'task_type' in the state and updates the state to route
    to the corresponding workflow node. It also clears the 'message' to prepare for
    the next step.
    """
    def __init__(self, llm):
        super().__init__("DataLoaderNode")
        self.llm = llm
        root_paths = ["/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B", "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3A"]
        with open("src/data/file_descriptions.json", "r") as file:
            valid_object_types = json.load(file).keys()

        self.index = self.index_simulation_directories(root_paths, valid_object_types)

    def run(self, state):
        """
        Execute the decision logic to determine the next workflow node.

        Args:
            state (dict): The current state dictionary containing at least 'task_type' and 'parameters'.
        """
    

    # def (self, question: str):
    #     llm_json = self.llm.bind(format="json")
    #     parser = JsonOutputParser(pydantic_object=ExtractedVariables)
    #     prompt = PromptTemplate(
    #         template= extract_template,
    #         input_variables=["question", "context"],
    #         partial_variables={"format_instructions": parser.get_format_instructions()},
    #     )
        
    #     context = self.retriever.invoke(question)
    #     response = (prompt | self.llm | parser).invoke({"question": question, "context": context})
    #     return response


    def index_simulation_directories(self, root_paths: List[str], valid_object_types: set) -> Dict[Tuple[float, float, float], Dict[int, Dict[str, str]]]:
        """
        Builds a hierarchical index from simulation analysis directories.
        Expected file structure:
        ROOT/FSN_{float}_VEL_{float}_TEXP_{float}/analysis/m000p-{timestep}.{object_type}

        Only object types in `valid_object_types` are included.
        """
        index = defaultdict(lambda: defaultdict(dict))

        float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        sim_pattern = re.compile(
            rf"FSN_{float_pattern}_VEL_{float_pattern}_TEXP_{float_pattern}_BETA_{float_pattern}_SEED_{float_pattern}"
        )
        file_pattern = re.compile(r"m000p-(\d+)\.(.+)$")

        for root_path in root_paths:
            for dirpath, _, filenames in os.walk(root_path):
                if os.path.basename(dirpath) != "analysis":
                    continue

                sim_match = sim_pattern.search(dirpath)
                if not sim_match:
                    continue

                sim_id = tuple(map(float, sim_match.groups()))

                for fname in filenames:
                    file_match = file_pattern.match(fname)
                    if not file_match:
                        continue

                    timestep = int(file_match.group(1))
                    object_type = file_match.group(2)

                    if object_type not in valid_object_types:
                        continue

                    full_path = os.path.join(dirpath, fname)
                    index[sim_id][timestep][object_type] = full_path

        return index