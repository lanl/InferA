import logging
from src.utils.config import NODE_VERBOSE, NODE_DEBUG

logger = logging.getLogger(__name__)

class NodeBase:
    def __init__(self, name: str):
        self.name = name
        self.debug = NODE_DEBUG
        self.verbose = NODE_VERBOSE
        self.logger = logger

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
