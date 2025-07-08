# main.py
# 
# Main entry point for LLM software

import os
import time
import threading
import uvicorn
import pickle

from src.utils.message_utils import pretty_print_message, pretty_print_messages
from src.core.llm_models import LanguageModelManager
from src.core.workflow import WorkflowManager
from src.llm import fastapi_client
from src.utils.logger_config import setup_logger

from src.utils.config import ENABLE_DEBUG, ENABLE_LOGGING, ENABLE_CONSOLE_LOGGING
from src.utils.config import WORKING_DIRECTORY, STATE_DICT_PATH

class MultiAgentSystem:
    def __init__(self, session_id: str = None):
        self.logger = setup_logger(ENABLE_LOGGING, ENABLE_DEBUG, ENABLE_CONSOLE_LOGGING)

        self.session_id = session_id
        self.state_dict = {}

        self.setup_environment()
        self.lm_manager = LanguageModelManager()
        self.workflow_manager = WorkflowManager(
            language_models = self.lm_manager.get_models(),
            working_directory = WORKING_DIRECTORY
        )

    def setup_environment(self):
        # Start FastAPI server in background
        server_thread = threading.Thread(target=self.setup_sandbox, daemon=True)
        server_thread.start()
        self.logger.info("FastAPI server running at http://127.0.0.1:8000")
        time.sleep(1)

        if not os.path.exists(WORKING_DIRECTORY):
            os.makedirs(WORKING_DIRECTORY)
            self.logger.info(f"Created working directory: {WORKING_DIRECTORY}")

        if not os.path.exists("./state/"):
            os.makedirs("./state/")
            self.logger.info(f"Created state storage directory: ./state/")
        
        # load previous state if it exists
        try:
            with open(STATE_DICT_PATH, 'rb') as file:
                self.state_dict = pickle.load(file)
                self.logger.info(f"[SESSION] Reading from previous session with ID: {self.session_id}")

        except:
            self.logger.info(f"[SESSION] {STATE_DICT_PATH} not found. Initializing from new state.")
            self.state_dict = {}


    def setup_sandbox(self):
        uvicorn.run("src.llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
        
        
    def run(self, user_input: str) -> None:
        session_id = self.session_id
        state_dict = self.state_dict
        config = {"configurable": {"thread_id": self.session_id}, "recursion_limit": 50}

        if session_id not in state_dict or not session_id:
            state_dict[session_id] = {}
            state_dict[session_id]['session_id'] = session_id
            state_dict[session_id]['messages'] = [user_input]
            state_dict[session_id]['user_inputs'] = [user_input]
            state_dict[session_id]['next'] = "Planner"
        else:
            self.logger.info(f"[SESSION] Starting previous session from node: {state_dict[session_id]['next']}")
            for k, v in state_dict[session_id].items():
                # Format the value nicely, you can customize this if v is complex
                if k in ["messages", "stashed_msg", "retrieved_docs", "db_columns"]:
                    continue
                if isinstance(v, (dict, list)):
                    import pprint
                    formatted_value = pprint.pformat(v, indent=4)
                else:
                    formatted_value = str(v)
                self.logger.info(f"  {k}: {formatted_value}")


        """Run multi-agent system with user input"""
        graph = self.workflow_manager.get_graph()

        try:
            with open("output/graph_output.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception as e:
            self.logger.warning(e)
        
        track_state = []
        max_history = 5

        try:
            for k, v in graph.stream(
                state_dict[session_id], config,
                stream_mode = ["values", "updates"]
            ):  
                try:
                    if k == "updates":
                        pretty_print_messages(v, last_message=True)
                    if k == "values":
                        track_state.append(v)
                except Exception as e:
                    self.logger.error(f"Unable to print message {k}:{v}. {e}")
                    print(f"Unable to print message {k}:{v}. {e}")
        except Exception as e:
            self.logger.error(e)
            print(e)    


        rewind_steps = 3
        if len(track_state) >= rewind_steps + 1:
            state_dict[session_id] = track_state[-(rewind_steps + 1)]
        else:
            # Not enough history to rewind that far, fallback to earliest or current
            state_dict[session_id] = track_state[0] if track_state else None

        # # write to disk
        # with open(STATE_DICT_PATH, 'wb') as file:
        #     pickle.dump(state_dict, file)
        #     self.logger.info(f"[SESSION] State saved to {STATE_DICT_PATH} using session ID = {self.session_id}")


def main():
    session_id = "138"
    # session_id = None
    system = MultiAgentSystem(session_id = session_id)
    # user_input = "Can you show me the change in mass of the largest friends-of-friends halos for all timesteps in simulation 0?"
    user_input = "What is the most important data point in the dataset?"
    # user_input = "Can you find me the largest friends-of-friends halo from timestep 498 in simulation 0?"
    # user_input = "Find me the 10 friends-of-friends halos closest in coordinates to the halo with fof_halo_tag = '251375070' in timestep 498 of simulation 0. Use columns 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z'."
    # user_input = "Can you map out the largest friends-of-friends halos for all timesteps in simulation 0? For visualization, instead of going to visualization agent, just have the python programmer provide code to visualize."
    system.run(user_input)


if __name__ == "__main__":
    main()
