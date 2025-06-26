import logging

logger = logging.getLogger(__name__)

class NodeBase:
    def __init__(self, name: str):
        self.name = name
        self.logger = logger

    def __call__(self, state):
        self.logger.info(f"[{self.name.upper()}] Starting node...")
        self.logger.debug(f"[{self.name.upper()}] Input state: {state}")

        result = self.run(state)

        self.logger.info(f"[{self.name.upper()}] Finished node.")
        self.logger.debug(f"[{self.name.upper()}] Output state: {result}")

        return result

    def run(self, state):
        raise NotImplementedError("Subclasses must implement run()")
