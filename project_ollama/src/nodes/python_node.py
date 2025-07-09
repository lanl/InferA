import re
import json
import logging
import duckdb
import pandas as pd

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser

from src.nodes.node_base import NodeBase
from src.core.fastapi_client import query_dataframe_agent

from src.utils.dataframe_utils import pretty_print_df, pretty_print_dict
from src.utils.config import WORKING_DIRECTORY



logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("Python")
        self.llm = llm
        self.generate_code = self._generate_code()
    
    def run(self, state):
        task = state["task"]
        session_id = state.get("session_id", "")
        db_path = state.get("db_path", None)
        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        # Aggregate all explanations for previous dfs
        explanations = self._aggregate_explanations(results_list)

        # Load dataframes for processing
        df = self._load_dataframe(results_list, db_path)
        if df is None:
            error_msg = "[PYTHON PROGRAMMER] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        columns = list(df.columns)
        try:
            response = self.generate_code.invoke({
                "task": task, 
                "columns": columns,
                "df_head": df.head().to_string(),
                "df_describe": df.describe().to_string(),
                "explanations": explanations,
                "df_types": df.dtypes.to_string()
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate code: {e}")


        python_code = self.extract_code_block(response.get("python_code", ""))

        explanation = response.get("explanation", "")
        logger.info(f"[PYTHON PROGRAMMER] Generated pandas code:\n{python_code}\nExplanation:\n{explanation}")
            
        # Execute the code safely from fastAPI server
        try:
            result = pd.DataFrame.from_dict(query_dataframe_agent(df, python_code))
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._error_response(f"Failed to execute code on server: {e}. Code: {python_code}.")
    
        return self._handle_result(result, python_code, df_index, results_list, explanation, session_id)


    def _load_dataframe(self, results_list, db_path):
        """
        Load dataframe either from CSV files in results_list or from a database.
        """
        if not results_list:
            logger.info(f"[PYTHON PROGRAMMER] No dataframes from previous steps. Getting dataframe from db.")       
            try:
                db = duckdb.connect(db_path)
                df = db.execute("SELECT * FROM data").fetchdf()
            except Exception as e:
                logger.error(f"Database load failed: {e}")
                return None

        else:
            try:
                dfs = []
                explanations = ""
                for path, _ in results_list:
                    if path.endswith(".csv"):
                        dfs.append(pd.read_csv(path))
                if not dfs:
                    logger.error("[PYTHON PROGRAMMER] No CSV files found in results_list.")
                    return None
                combined_df = pd.concat(dfs)
                logger.info(f"[PYTHON PROGRAMMER] Loaded and concatenated CSV dataframes.")
                # pretty_print_df(combined_df)
                return combined_df

            except Exception as e:
                logger.error(f"[PYTHON PROGRAMMER] Failed to load CSV: {str(e)}")
                return None
    

    def _aggregate_explanations(self, results_list):
        """
        Aggregate explanations from previous results.
        Handles cases where explanation is a dict instead of a string.
        """
        explanation_texts = []
        for _, explanation in results_list:
            if isinstance(explanation, dict):
                try:
                    explanation_texts.append(json.dumps(explanation, indent=2))
                except Exception:
                    explanation_texts.append(str(explanation))
            else:
                explanation_texts.append(str(explanation))
        return "\n".join(explanation_texts)
    

    def _generate_code(self):
        pandas_schema = [
            ResponseSchema(
                name="python_code", 
                description="Python code using pandas to process or analyze the input DataFrame, input_df."
            ),
            ResponseSchema(
                name="explanation", 
                description="Brief explanation of what the code is doing."
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(pandas_schema)

        system_prompt = (
            """
            You are a python coding assistant with expertise in working with the Pandas library.
            Your task is to transform a pandas DataFrame named `input_df` based on user instructions. 

            ***STRICT RULES (Always Follow These):***
            - ❌ NEVER import any libraries (no `import pandas`, `import numpy`, etc.).
            - ✅ ALWAYS return a single Python code block using triple backticks: ```python ...code... ```
            - ✅ ALWAYS assign the final result to a single DataFrame named `result`.
            - ❌ NEVER use loops, file I/O, or print statements.
            - ✅ Use only pandas and numpy operations.
            - ✅ If the user’s task applies to only part of the DataFrame, return just the relevant rows or columns.

            ***Incorrect example (Do NOT do this):***
            ``` # Must add python code block
            import pandas as pd # Do not import libraries
            
            result = ...
            ```

            First, think step-by-step about how to perform the task using pandas/numpy.
            Then, return only the final code inside a Python code block.

            **Task:** 
            {task}

            **Columns in dataframe:**
            {columns}

            **DataFrame (first few rows):**
            ```
            {df_head}
            ```

            **DataFrame Statistical Summary (`df.describe()`):**
            ```
            {df_describe}
            ```

            **DataFrame types:**
            {df_types}

            **Previous steps in the analysis pipeline:**
            {explanations}

            {format_instructions}

            Respond only with JSON.
            """
        )
        
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "columns", "df_head", "df_describe", "explanations", "df_types"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        return prompt_template | self.llm | output_parser
    

    def _handle_result(self, result, python_code, df_index, results_list, explanation, session_id):
        """
        Process the result returned by executing the generated pandas code.
        """
        if isinstance(result, dict) and "error_type" in result and "error_message" in result:
                error_str = f"Execution returned error: {result['error_type']}: {result['error_message']}.\nCode: {python_code}"
                return self._error_response(error_str)
        
        elif isinstance(result, pd.DataFrame):
            return_file = f"{WORKING_DIRECTORY}{session_id}_{df_index}.csv"
            logger.info(f"\033[44m[PYTHON PROGRAMMER] Writing dataframe to {return_file}\033[0m")
            result.to_csv(return_file, index=False)
            df_index += 1
            results_list.append((return_file, explanation))
            pretty_output = pretty_print_df(result, return_output=True)

        elif isinstance(result, dict):
            df_index += 1
            results_list.append((f"python_{df_index}", result))
            pretty_output = pretty_print_dict(result, return_output=True)
        else:
            pretty_output = str(result)
        stashed_msg = f"Python code\n{python_code}\n\nOutput:\n{pretty_output}"
        return {
            "next": "QA",
            "current": "PythonProgrammer",
            "messages": [AIMessage(f"Pandas code executed successfully.\n\n{pretty_output}")],
            "stashed_msg": stashed_msg,
            "results_list": results_list,
            "df_index": df_index,
        }
    

    def _error_response(self, message):
        """
        Helper for consistent error response format.
        """
        logger.error(f"\033[1;31m{message}\033[0m")
        return {
            "next": "QA",
            "current": "PythonProgrammer",
            "messages": [AIMessage(message)],
            "stashed_msg": message,
        }
    

    def extract_code_block(self, code_str: str) -> str:
        """
        Extract Python code inside triple backticks ```python ... ```
        Returns cleaned code string.
        """
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        # fallback: strip any triple backticks if present
        return code_str.strip().strip("```").strip()