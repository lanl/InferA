# main.py
# 
# Main entry point for LLM software

import os
import time
import threading
import uvicorn
import pickle

from src.core.message_utils import pretty_print_message, pretty_print_messages
from src.core.llm_models import LanguageModelManager
from src.core.workflow import WorkflowManager
from src.llm import fastapi_client
from src.utils.logger_config import setup_logger

from src.utils.config import ENABLE_DEBUG, ENABLE_LOGGING, ENABLE_CONSOLE_LOGGING
from src.utils.config import WORKING_DIRECTORY, STATE_DICT_PATH

class MultiAgentSystem:
    def __init__(self, session_id: str = None):
        self.logger = setup_logger(ENABLE_LOGGING, ENABLE_DEBUG, ENABLE_CONSOLE_LOGGING)
        self.setup_environment()
        self.lm_manager = LanguageModelManager()
        self.workflow_manager = WorkflowManager(
            language_models = self.lm_manager.get_models(),
            working_directory = WORKING_DIRECTORY
        )

        self.session_id = session_id
        self.state_dict = {}

    def setup_environment(self):
        # Start FastAPI server in background
        server_thread = threading.Thread(target=self.setup_sandbox, daemon=True)
        server_thread.start()
        self.logger.info("FastAPI server running at http://127.0.0.1:8000")
        time.sleep(1)

        if not os.path.exists(WORKING_DIRECTORY):
            os.makedirs(WORKING_DIRECTORY)
            self.logger.info(f"Created working directory: {WORKING_DIRECTORY}")
        
        # load previous state if it exists
        try:
            with open(STATE_DICT_PATH, 'rb') as file:
                self.state_dict = pickle.load(file)
        except:
            self.state_dict = {}


    def setup_sandbox(self):
        uvicorn.run("src.llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
        
        
    def run(self, user_input: str) -> None:
        session_id = self.session_id
        state_dict = self.state_dict
        config = {"configurable": {"thread_id": "1"}}

        # check for previous state
        if session_id in state_dict:
            state_dict[session_id]['messages'].append(user_input)
        else:
            state_dict[session_id] = {}
            state_dict[session_id]['session_id'] = session_id
            state_dict[session_id]['messages'] = [user_input]
            state_dict[session_id]['user_inputs'] = [user_input]

        """Run multi-agent system with user input"""
        graph = self.workflow_manager.get_graph()

        try:
            with open("output/graph_output.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception:
            pass

        for s in graph.stream(
            state_dict[session_id], config,
            stream_mode = "updates"
        ):  
            pretty_print_messages(s, last_message=True)

        # # update state
        # state_dict[session_id] = s[key]

        # # write to disk
        # with open(STATE_DICT_PATH, 'wb') as file:
        #     pickle.dump(state_dict, file)
        
        # # collect response
        # last_message = s[key]['messages'][-1].content
        
        # return last_message


def main():
    system = MultiAgentSystem()
    user_input = "I want to find the largest halo for timestep 498 in simulation 1."
    system.run(user_input)


if __name__ == "__main__":
    main()
