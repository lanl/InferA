from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseWorkflow(ABC):
    @abstractmethod
    def run(self, **kwargs):
        """Run the workflow with given parameters"""
        pass