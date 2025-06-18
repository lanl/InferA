import logging
from src.utils.logger_config import get_logger
from src.utils.config import NODE_VERBOSE, NODE_DEBUG

class NodeBase:
    def __init__(self, name: str):
        self.name = name
        self.debug = NODE_DEBUG
        self.verbose = NODE_VERBOSE
        self.logger = get_logger(self.name)

    def _log(self, message: str):
        self.logger.info(message)

    def __call__(self, state):
        if self.verbose:
            self._log(f"[→] Starting node: {self.name}")
        if self.debug:
            self._log(f"[{self.name}] Input state: {state}")

        result = self.run(state)

        if self.verbose:
            self._log(f"[→] Finished node: {self.name}")
        if self.debug:
            self._log(f"[{self.name}] Output state: {result}")

        return result

    def run(self, state):
        raise NotImplementedError("Subclasses must implement run()")
