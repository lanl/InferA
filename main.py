# main.py
# 
# Main entry point for LLM software

import os
import time
import json
import threading
import uvicorn
import pickle

import atexit
import time

from src.utils.message_utils import pretty_print_messages
from src.core.llm_models import LanguageModelManager
from src.core.workflow import WorkflowManager
from src.utils.logger_config import setup_logger

from config import WORKING_DIRECTORY, STATE_DICT_PATH, PRINT_DEBUG_TO_CONSOLE


class MultiAgentSystem:
    def __init__(self, session: str = None, step: str = "0"):
        self.start_time = time.time()
        self.end_time = None
        self.logger = setup_logger(print_debug_to_console=PRINT_DEBUG_TO_CONSOLE)

        self.session = session
        self.step = step
        self.state_key = f"{self.session}_{self.step}"
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
        self.logger.info("FastAPI server running at http://127.0.0.1:3000")
        time.sleep(1)

        if not os.path.exists(WORKING_DIRECTORY):
            os.makedirs(WORKING_DIRECTORY)
            self.logger.debug(f"Created working directory: {WORKING_DIRECTORY}")
        
        if not os.path.exists(f"{WORKING_DIRECTORY}{self.session}/"):
            os.makedirs(f"{WORKING_DIRECTORY}{self.session}/")
            self.logger.debug(f"Created state file storage directory: {WORKING_DIRECTORY}{self.session}/")

        if not os.path.exists("./state/"):
            os.makedirs("./state/")
            self.logger.debug(f"Created state dictionary directory: ./state/")
        
        
        # load previous state if it exists
        try:
            with open(STATE_DICT_PATH, 'rb') as file:
                self.state_dict = pickle.load(file)
                self.logger.info(f"[SESSION] Reading from previous session with ID: {self.session} at step {self.step}")

        except:
            self.logger.info(f"[SESSION] {STATE_DICT_PATH} not found. Initializing from new state.")
            self.state_dict = {}


    def setup_sandbox(self):
        uvicorn.run("src.core.fastapi_server:app", host="127.0.0.1", port=3000, reload=False, log_level="info")


    def cleanup(self):
        if self.step == "-1":
            self.logger.info(f"[SESSION] Steps set to '-1'. Disabled saving.")
        else:
            # Save readable log
            with open(STATE_DICT_PATH, "wb") as f:
                pickle.dump(self.state_dict, f)
                self.logger.info(f"[SESSION] All streamed steps saved to {STATE_DICT_PATH}.")
        
        self.end_time = time.time()
        self.logger.info(f"Total time server was running - took {self.end_time - self.start_time:.4f} seconds to run.")
        
        
    def run(self, user_input: str) -> None:
        state_key = self.state_key
        config = {"configurable": {"thread_id": state_key}, "recursion_limit": 75}

        if state_key not in self.state_dict:
            self.state_dict[state_key] = {
                "session_id": self.session,
                "state_key": state_key,
                "messages": [user_input],
                "user_inputs": [user_input],
                "next": "Planner"
            }
        else:
            self.logger.info(f"[SESSION] Starting previous session from node: {self.state_dict[state_key]['next']}")

            # Print entire state dictionary
            for k, v in self.state_dict[state_key].items():
                # Format the value nicely, you can customize this if v is complex
                if k in ["messages", "user_input", "stashed_msg", "retrieved_docs", "db_columns", "file_index"]:
                    self.logger.debug(f"  {k}: {v}")
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
                self.state_dict[state_key], config,
                stream_mode = ["values", "updates"]
            ):  
                try:
                    if k == "updates":
                        pretty_print_messages(v, last_message=True)
                    if k == "values":
                        # Save each step as a new key in state_dict
                        new_state_key = f"{self.session}_{step_counter}"
                        self.state_dict[new_state_key] = v

                        step_counter += 1

                        if self.step != -1:
                            self.logger.info(f"State saved to : {new_state_key}.\n")
                        else:
                            self.logger.info(f"Step save disabled.\n")

                except Exception as e:
                    self.logger.error(f"Unable to print message {k}:{v}. {e}")
        except Exception as e:
            self.logger.error(f"Graph stream failed: {e}")


def main():
    session = "11001"
    step = "0"
    system = MultiAgentSystem(session = session, step = step)

    question_id = 11

    with open("src/data/example_questions.json", "r", encoding='utf-8') as f:
        data = json.load(f)    
    
    user_input = ""
    for question in data.get("questions", []):
        if question.get("id") == question_id:
            user_input = question.get("text")

    system.run(user_input)



if __name__ == "__main__":
    main()
