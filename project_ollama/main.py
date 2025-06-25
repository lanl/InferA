# main.py
# 
# Main entry point for LLM software

import os
import time
import threading
import uvicorn

from langchain_core.messages import HumanMessage
from IPython.display import Image, display

from src.core.message_utils import pretty_print_message, pretty_print_messages
from src.core.llm_models import LanguageModelManager
from src.core.workflow import WorkflowManager
from src.llm import fastapi_client
from src.utils.logger_config import setup_logger

from src.utils.config import ENABLE_DEBUG, ENABLE_LOGGING, ENABLE_CONSOLE_LOGGING
from src.utils.config import WORKING_DIRECTORY

class MultiAgentSystem:
    def __init__(self):
        self.logger = setup_logger(ENABLE_LOGGING, ENABLE_DEBUG, ENABLE_CONSOLE_LOGGING)
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

    def setup_sandbox(self):
        uvicorn.run("src.llm.fastapi_server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
        
        
    def run(self, user_input: str) -> None:
        """Run multi-agent system with user input"""
        graph = self.workflow_manager.get_graph()

        try:
            with open("output/graph_output.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception:
            pass

        for chunk in graph.stream(
            {"messages": [("user", user_input)]},
            subgraphs = True
        ):
            pretty_print_messages(chunk, last_message=True)


def main():
    system = MultiAgentSystem()
    user_input = "I want to find the largest halo from simulation 1."
    system.run(user_input)


if __name__ == "__main__":
    main()
