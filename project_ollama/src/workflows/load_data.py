import sys
import pandas as pd
import numpy as np

from src.utils import json_loader
from src.workflows.base_workflow import BaseWorkflow

genericio_path = "genericio/legacy_python/"
sys.path.append(genericio_path)

import genericio as gio


class DataLoader(BaseWorkflow):
    def __init__(self, base_path):
        self.base_path = base_path

    def read_gio_to_df(self, object_type: str, timestep: int, file_type: str) -> pd.DataFrame:
        file = f"{self.base_path}/m000p-{timestep}.{object_type}{file_type}"
        vars = json_loader.get_variable_names_from_json(f"{object_type}{file_type}")
        values = gio.read(file, vars)
        return pd.DataFrame(np.column_stack(values), columns=vars)
