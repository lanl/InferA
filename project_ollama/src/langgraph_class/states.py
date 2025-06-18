from typing_extensions import TypedDict
from typing import Optional, Dict, Any, List

class AnalysisState(TypedDict, total = False):
    # 1. Preprocessing step to get all variables
    base_dir: str
    full_dir: str
    next_node: Optional[str]
    exit: Optional[bool]
    data_variables: Optional[Dict[str, Any]] = None
    preprocessing_complete: bool = False

    # 2. Feedback loop for user input
    user_input: Optional[str]
    task_type: Optional[str]
    parameters: Dict[str, Any]
    missing_parameters: List[str]
    message: List[str]
    use_visual: Optional[bool]

    # 3. Once a task has been decided and parameters collected, call workflow
    result: Optional[Any]
    visual_output: Optional[str]
    track_evolution: Optional[bool]
    evolution_result: Optional[Any]

    # 4. Workflow has completed - prompt user for further analysis
    start_visual: Optional[bool]
    start_explorer: Optional[bool]
    retriever: Optional[Any]

