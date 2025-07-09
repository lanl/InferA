# main.py
# 
# Main entry point for LLM software

import os
import time
import threading
import uvicorn
import pickle

import atexit

from src.utils.message_utils import pretty_print_messages
from src.core.llm_models import LanguageModelManager
from src.core.workflow import WorkflowManager
from src.utils.logger_config import setup_logger

from src.utils.config import ENABLE_DEBUG, ENABLE_LOGGING, ENABLE_CONSOLE_LOGGING
from src.utils.config import WORKING_DIRECTORY, STATE_DICT_PATH


class MultiAgentSystem:
    def __init__(self, session: str = None, step: str = "0"):
        self.logger = setup_logger(ENABLE_LOGGING, ENABLE_DEBUG, ENABLE_CONSOLE_LOGGING)

        self.session = session
        self.step = step
        self.state_dict = {}

        self.setup_environment()
        self.lm_manager = LanguageModelManager()
        self.workflow_manager = WorkflowManager(
            language_models = self.lm_manager.get_models(),
            working_directory = WORKING_DIRECTORY
        )

        atexit.register(self.cleanup)

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
                self.logger.info(f"[SESSION] Reading from previous session with ID: {self.session} at step {self.step}")
                print(f"[SESSION] Reading from previous session with ID: {self.session} at step {self.step}")

        except:
            self.logger.info(f"[SESSION] {STATE_DICT_PATH} not found. Initializing from new state.")
            self.state_dict = {}


    def setup_sandbox(self):
        uvicorn.run("src.core.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


    def cleanup(self):
        # Save readable log
        with open(STATE_DICT_PATH, "wb") as f:
            pickle.dump(self.state_dict, f)
            self.logger.info(f"[SESSION] All streamed steps saved to {STATE_DICT_PATH}.")
            print(f"[SESSION] All streamed steps saved to {STATE_DICT_PATH}.")
        
        
    def run(self, user_input: str) -> None:
        session_id = f"{self.session}_{self.step}"
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

        if session_id not in self.state_dict:
            self.state_dict[session_id] = {
                "session_id": self.session,
                "step": self.step,
                "messages": [user_input],
                "user_inputs": [user_input],
                "next": "Planner"
            }
        else:
            self.logger.info(f"[SESSION] Starting previous session from node: {self.state_dict[session_id]['next']}")

            # Print entire state dictionary
            for k, v in self.state_dict[session_id].items():
                # Format the value nicely, you can customize this if v is complex
                if k in ["messages", "user_input", "stashed_msg", "retrieved_docs", "db_columns", "file_index"]:
                    continue
                if k in ["results_list"]:
                    values = ', '.join(str(i[0]) for i in v)
                    self.logger.info(f"  {k}: {values}")
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
            with open("state/graph_output.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception as e:
            self.logger.warning(e)
        
        # Collects every (k,v) pair for writing to save state.
        track_states = {}
        step_counter = int(self.step)

        try:
            for k, v in graph.stream(
                self.state_dict[session_id], config,
                stream_mode = ["values", "updates"]
            ):  
                try:
                    if k == "updates":
                        if not ENABLE_CONSOLE_LOGGING:
                            pretty_print_messages(v, last_message=True)
                        self.logger.info(pretty_print_messages(v, last_message=True))
                    if k == "values":
                        # Save each step as a new key in state_dict
                        state_key = f"{self.session}_{step_counter}"
                        self.state_dict[state_key] = v

                        step_counter += 1
                        self.logger.info(f"State saved to : {state_key}.")
                        print(f"State saved to : {state_key}.")

                except Exception as e:
                    self.logger.error(f"Unable to print message {k}:{v}. {e}")
                    print(f"Unable to print message {k}:{v}. {e}")
        except Exception as e:
            self.logger.error(e)
            print(e)


def main():
    session = "123"
    step = "29"
    # session_id = None
    system = MultiAgentSystem(session = session, step = step)

    # user_input = input("You: ")

    # user_input = "Can you help me plot the change in mass of the largest friends-of-friends halos for all timesteps in simulation 0?"
    # user_input = "Can you find me the top 20 largest friends-of-friends halos from timestep 498 in simulation 0?"
    user_input = "I want to visualize the top 20 largest friends-of-friends halos from timestep 498 in simulation 0?"
    # user_input = "Find me the 10 friends-of-friends halos closest in coordinates to the halo with fof_halo_tag = '251375070' in timestep 498 of simulation 0. Use columns 'fof_halo_tag', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z'."
    # user_input = "Can you map out the largest friends-of-friends halos for all timesteps in simulation 0?"
    # user_input = "What are the top 10 largest galaxies in halo with fof_halo_tag = '251375070' in timestep 498 of simulation 0?"
    # user_input = ""
    # user_input = "Can you plot a vtk of all the halos within 10 Mpc of the largest halo at timestep 498 of simulation 0?"
    # user_input = "Can you plot a timeseries pvd for all the largest halos at every timestep in simulation 0? At each timestep also include every halo within a distance of 10 Mpc."
    # user_input = "Visualize all halos within 10 Mpc of the coordinate (20,20,20) for timestep 498 of simulation 0."
    
    system.run(user_input)



if __name__ == "__main__":
    main()
