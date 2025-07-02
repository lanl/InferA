import logging
import duckdb
import pandas as pd

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser

from src.langgraph_class.node_base import NodeBase

from src.llm.fastapi_client import query_dataframe_agent
from src.utils.json_loader import extract_code_block

from src.utils.dataframe_utils import pretty_print_df



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
        df_path = state.get("df_path", [])
        df_index = state.get("df_index", 0)

        # if not df_path:
        #     logger.info(f"[SQL PROGRAMMER] Database has not been written. Routing back to supervisor.")       
        #     return {"next": "Supervisor", 
        #             "current": "SQLProgrammer", 
        #             "messages": [AIMessage("Database is missing. Check with DataLoader to verify.")]
        #         }
        df_list = []
        try:
            for path in df_path:
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
                "stashed_msg": error_msg
            }
        
        logger.info(f"[PYTHON PROGRAMMER] Task: {task}")
        try:
            columns = list(df.columns)
            response = self.generate_pandas.invoke({"task": task, "columns": columns})

            unfiltered_code = response["pandas_code"]
            pandas_code = extract_code_block(unfiltered_code)

            explanation = response["explanation"]
            logger.info(f"[PYTHON PROGRAMMER] Generated pandas code:\n{pandas_code}\n\n")
            print(f"Explanation:\n{explanation}\n\n")

            if not pandas_code.startswith("python"):
                raise ValueError("Generated code does not start with 'df', refusing to execute for safety.")
            else:
                pandas_code = pandas_code[len("python"):].strip()
            
             # Execute the code safely from fastAPI server
            try:
                result = query_dataframe_agent(df, pandas_code)
            except Exception as e:
                logger.error(f"Execution error: {e}")


            if isinstance(result, pd.DataFrame):
                pretty_print_df(result)
                # Save filtered dataframe to new CSV
                output_csv = f"{df_index}.csv"
                result.to_csv(output_csv, index=False)
                df_path.append(output_csv)
                df_index += 1
                return {
                    "next": "QA",
                    "current": "PythonProgrammer",
                    "messages": [AIMessage(f"Pandas code executed successfully.\nCode:\n{pandas_code}")],
                    "stashed_msg": f"Pandas code:\n{pandas_code}",
                    "df_path": df_path,
                    "df_index": df_index,
                    "result": output_csv
                }
            else:
                # If result is scalar or other type, just return as message
                return {
                    "next": "QA",
                    "current": "PythonProgrammer",
                    "messages": [AIMessage(f"Result:\n{result}")],
                    "stashed_msg": f"Result:\n{result}",
                    "result": result
                }

        except Exception as e:
            error_msg = f"[PYTHON PROGRAMMER] Error executing pandas code: {str(e)}"
            # logger.error(error_msg)
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
                description="Python code using pandas to process or analyze the input DataFrame, df."
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

            Requirements:
            - Only use pandas and numpy. 
            - If the rest of the dataset is not relevant to the user's query, only return the relevant columns or rows.
            - Do not import any libraries, including pandas or numpy.
            - Do not use loops, file I/O, or print statements.
            - Assign the final result to a new DataFrame named `result_df`.
            - You should think through the steps for how to best answer the question before generating code.
            - Return only the code, inside a single Python code block (```python ...```).
            - The following context contains column names in 'input_df', use this as context for which columns to transform.
            
            Columns in dataframe:
            {columns}

            Task: 
            {task}

            {format_instructions}

            Respond only with JSON.
            """
        )
        
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "columns"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        return prompt_template | self.llm | output_parser
