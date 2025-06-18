# main.py
# 
# Main entry point for LLM software

from src.llm.llm_client import llm, embedding_model
from src.llm import fastapi_client
from src.utils.logger_config import get_logger

import time
import threading
import uvicorn

# from IPython.display import display, Image

from src.langgraph_class.states import AnalysisState
from src.langgraph_class.graph_builder import build_graph


datapath_flamingo_B_1 = "/vast/projects/exasky/data/hacc/SCIDAC_RUNS/128MPC_RUNS_FLAMINGO_DESIGN_3B/FSN_0.5387_VEL_149.279_TEXP_9.613_BETA_0.8710_SEED_9.548e5"
flamingo_B_1_analysis = datapath_flamingo_B_1 + "/analysis"
flamingo_B_1_output = datapath_flamingo_B_1 + "/output"


def start_fastapi():
    uvicorn.run("src.llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


def main():
    logging = get_logger(__name__)

    # Start FastAPI server in background
    server_thread = threading.Thread(target=start_fastapi, daemon=True)
    server_thread.start()
    logging.info("FastAPI server running at http://127.0.0.1:8000")
    time.sleep(1)


    initial_state: AnalysisState = {
        "base_dir": flamingo_B_1_analysis,
        "full_dir": flamingo_B_1_output,
    }

    # # Testing workflow initial state:
    # initial_state: AnalysisState = {
    #     "base_dir": flamingo_B_1_analysis,
    #     "full_dir": flamingo_B_1_output,
    #     "task_type": "find_largest_within_halo",
    #     "use_visual": False,
    #     "parameters": {'object_type': 'galaxy', 'timestep': 498, 'halo_id': 226693946.0	, 'n': 2}
    # }

    # # Testing workflow initial state:
    # initial_state: AnalysisState = {
    #     "base_dir": flamingo_B_1_analysis,
    #     "full_dir": flamingo_B_1_output,
    #     "task_type": "find_largest_object",
    #     "use_visual": True,
    #     "parameters": {'object_type': 'halo', 'timestep': 445, 'n': 5}
    # }

    # # Testing workflow initial state:
    # initial_state: AnalysisState = {
    #     "base_dir": flamingo_B_1_analysis,
    #     "full_dir": flamingo_B_1_output,
    #     "task_type": "track_object_evolution",
    #     "use_visual": False,
    #     "parameters": {'object_type': 'halo', 'object_id': 205888532.0, 'timestep': 498}
    # }

    graph = build_graph(llm, embedding_model)
    with open("output/graph_output.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    final_state = graph.invoke(initial_state)


if __name__ == "__main__":
    main()
