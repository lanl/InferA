import logging
import duckdb
import pandas as pd
from typing import Dict

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser

from src.langgraph_class.node_base import NodeBase
from src.utils.config import WORKING_DIRECTORY

from src.llm.fastapi_client import query_dataframe_agent
from src.utils.json_loader import extract_code_block

from src.utils.dataframe_utils import pretty_print_df, pretty_print_dict



logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("Python")
        self.llm = llm
        self.generate_pandas = self._generate_pandas()
    
    def run(self, state):
        task = state["task"]
        
        # Get db information in case SQL not run and not dataframes in state
        db_path = state.get("db_path", None)
        object_type = state.get("object_type", None)
        db_columns = state.get("db_columns", None)

        # Get dataframes in state, primary way to use pandas data
        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        if not results_list:
            logger.info(f"[PYTHON PROGRAMMER] No dataframes from previous steps. Getting dataframe from db.")       
            db = duckdb.connect(db_path)
            df = db.execute("SELECT * FROM data").fetchdf()

        else:
            try:
                df_list = []
                explanations = ""
                for result in results_list:
                    path = result[0]
                    explanations = explanations + "\n" + result[1]
                    if path.endswith(".csv"):
                        temp_df = pd.read_csv(path)
                        df_list.append(temp_df)
                df = pd.concat(df_list)
                logger.info(f"[PYTHON PROGRAMMER] Loaded and concatenated all dataframes to df.")
                pretty_print_df(df)

            except Exception as e:
                error_msg = f"[PYTHON PROGRAMMER] Failed to load CSV: {str(e)}"
                logger.error(error_msg)
                return {
                    "next": "QA",
                    "current": "PythonProgrammer",
                    "messages": [AIMessage(error_msg)],
                    "stashed_msg": error_msg,
                }
        
        logger.info(f"[PYTHON PROGRAMMER] Task: {task}")
        try:
            columns = list(df.columns)
            response = self.generate_pandas.invoke({
                "task": task, 
                "columns": columns,
                "df_head": df.head(),
                "df_describe": df.describe(),
                "explanations": explanations,
                "df_types": df.types
            })

            unfiltered_code = response["pandas_code"]
            pandas_code = extract_code_block(unfiltered_code)

            explanation = response["explanation"]
            logger.info(f"[PYTHON PROGRAMMER] Generated pandas code:\n{pandas_code}\n\n")
            print(f"Explanation:\n{explanation}\n\n")

            if not pandas_code.startswith("python"):
                # raise ValueError("Generated code does not start with 'python', refusing to execute for safety.")
                pass
            else:
                pandas_code = pandas_code[len("python"):].strip()
            
             # Execute the code safely from fastAPI server
            try:
                result = query_dataframe_agent(df, pandas_code)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                return {
                    "next": "QA",
                    "current": "PythonProgrammer",
                    "messages": [AIMessage(e)],
                    "stashed_msg": e
                }


            if isinstance(result, pd.DataFrame):
                pp_df = pretty_print_df(result, return_output = True)
                # Save filtered dataframe to new CSV
                return_result = f"{WORKING_DIRECTORY}{df_index}.csv"

                logger.info(f"\033[1;33m[PYTHON PROGRAMMER] Writing dataframe result to {return_result}.\033[0m")
                result.to_csv(return_result, index=False)
                
                df_index += 1
                results_list.append((return_result, explanation))

            elif isinstance(result, Dict):
                pp_df = pretty_print_dict(result, return_output = True)
                return_result = result
                
                df_index += 1
                results_list.append((f"python_{df_index}", return_result))
            
            return {
                "next": "QA",
                "current": "PythonProgrammer",
                "messages": [AIMessage(f"Pandas code executed successfully.\n\n{pp_df}")],
                "stashed_msg": f"Pandas code:\n{pandas_code}",
                "results_list": results_list,
                "df_index": df_index,
            }

        except Exception as e:
            error_msg = f"❌ \033[1;31m[PYTHON PROGRAMMER] Error executing pandas code: {str(e)}\033[0m"
            return {
                "next": "QA",
                "current": "PythonProgrammer",
                "messages": [AIMessage(error_msg)],
                "stashed_msg": error_msg
            }
        

    def _generate_pandas(self):
        pandas_schema = [
            ResponseSchema(
                name="pandas_code", 
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
            You are a code generator that transforms a pandas DataFrame named `input_df` based on user instructions. 

            ***STRICT RULES (Always Follow These):***
            - ❌ NEVER import any libraries (no `import pandas`, `import numpy`, etc.).
            - ✅ ALWAYS return a single Python code block using triple backticks: ```python ...code... ```
            - ✅ ALWAYS assign the final result to a single DataFrame named `result_df`.
            - ❌ NEVER use loops, file I/O, or print statements.
            - ✅ Use only pandas and numpy operations.
            - ✅ If the user’s task applies to only part of the DataFrame, return just the relevant rows or columns.

            ***Incorrect example (Do NOT do this):***
            ``` # Must add python code block
            import pandas as pd # Do not import libraries
            
            result_df = ...
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

            **Previous steps from previous steps in the analysis pipeline:**
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