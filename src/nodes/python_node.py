import os
import logging
import pandas as pd

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate

from src.nodes.node_base import NodeBase
from src.core.fastapi_client import query_dataframe_agent
from src.utils.dataframe_utils import pretty_print_df, pretty_print_dict

from config import WORKING_DIRECTORY


logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, code_tools, load_tools):
        super().__init__("Python")
        self.load_llm = llm.bind_tools(load_tools, parallel_tool_calls = False)
        self.llm = llm.bind_tools(code_tools, parallel_tool_calls = False)
        self.call_load = self._call_load()
        self.call_tool = self._call_tool()
        self.session_id = None
    
    def run(self, state):
        task = state["task"]
        self.session_id = state.get("session_id", "")

        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        print(results_list)
        # Aggregate all explanations for previous dfs
        output_descriptions= self._aggregate_results_description(results_list)

        try:
            response = self.call_load.invoke({
                "task": task,
                "output_descriptions": output_descriptions
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate tool.\nError:\n{e}")
        
        print(response)
        
        tool_calls = response.tool_calls
        data_paths = tool_calls[0]['args']['data_paths']
        logger.info(f"[PYTHON PROGRAMMER] To best answer the task, using data from {data_paths}")

        # Load dataframes for processing
        dfs = self._load_dataframe(data_paths)
        if dfs is None:
            error_msg = "[PYTHON PROGRAMMER] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        # Create a comprehensive description of all dataframes
        all_dfs_info = []
        for i, df in enumerate(dfs):
            df_info = f"DataFrame {i+1}:\n"
            df_info += f"Columns: {list(df.columns)}\n"
            df_info += f"Head:\n{df.head().to_string()}\n"
            df_info += f"Description:\n{df.describe().to_string()}\n"
            df_info += f"Data Types:\n{df.dtypes.to_string()}\n"
            df_info += "-" * 50 + "\n"  # Separator between dataframes
            all_dfs_info.append(df_info)

        # Join all dataframe descriptions into a single string
        all_dfs_description = "\n".join(all_dfs_info)

        try:
            response = self.call_tool.invoke({
                "task": task, 
                "dataframes_info": all_dfs_description,
                "num_dataframes": len(dfs)
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate tool.\nError:\n{e}")
        
        tool_calls = response.tool_calls[0]
        if not tool_calls:
            logger.error(f"No tools called.")
            return self._error_response(f"No tools called. Must return at least one tool.")
        
        logger.debug(f"[PYTHON PROGRAMMER] Tools called: {tool_calls}")
        
        if tool_calls.get("name") != 'GenerateCode':
            return {"messages": [response], "next": "PythonTool", "current": "PythonProgrammer"}

        # code_response = [t for t in tool_calls if t.get("name") == 'GenerateCode']
        # other_tools = [t for t in tool_calls if t.get("name") != 'GenerateCode']

        # if other_tools:
        #     return {"messages": other_tools, "next": "PythonTool", "current": "PythonProgrammer"}
        elif tool_calls.get("name") == 'GenerateCode':
            code_response = tool_calls

            import_code = code_response['args']['imports']
            python_code = code_response['args']['python_code']
            result_output_code = code_response['args']['result_output_code']
            explanation = code_response['args']['explanation']
            output_description = code_response['args']['output_description']

            python_code = python_code + "\n" + result_output_code

            logger.debug(f"\033[1;30;47m[PYTHON PROGRAMMER] Imports:\n\n{import_code}\n\nGenerated code:\n\n{python_code}\n\nExplanation:\n{explanation}\033[0m\n\n")
            # Execute the code safely from fastAPI server
            try:
                result = query_dataframe_agent(dfs, python_code, imports=import_code)
                
                # Handle different result types
                if isinstance(result, pd.DataFrame):
                    # Result is already a DataFrame, no conversion needed
                    pass
                elif isinstance(result, pd.Series):
                    # Convert Series to DataFrame
                    result = result.to_frame()
                elif result is None:
                    # Handle None result
                    result = pd.DataFrame()  # Empty DataFrame or appropriate message
                elif isinstance(result, str):
                    # If result is a string, you might want to display it differently
                    # or convert it to a DataFrame with a single cell
                    result = pd.DataFrame({'result': [result]})
                else:
                    # For other types, convert to string and then to DataFrame
                    result = pd.DataFrame({'result': [str(result)]})
                
            except Exception as e:
                return self._error_response(f"Failed to execute code on server\n\nError:{e}\n\nCode: {import_code}\n{python_code}")
        
            return self._handle_result(result, import_code, python_code, explanation, output_description, df_index)


    def _load_dataframe(self, data_path):
        """
        Load dataframe either from CSV files in results_list or from a database.
        """
        dfs = []
        try:
            for path in data_path:
                if path.endswith(".csv"):
                    dfs.append(pd.read_csv(path))
            if not dfs:
                logger.error(f"No CSV files found in results_list: {data_path}.")
                return None
            # combined_df = pd.concat(dfs)
            logger.debug(f"[PYTHON PROGRAMMER] Loaded and concatenated CSV dataframes.")
            return dfs

        except Exception as e:
            logger.error(f"[PYTHON PROGRAMMER] Failed to load CSV from {data_path}: {str(e)}")
            return None
    

    def _aggregate_results_description(self, results_list):
        """
        Aggregate explanations from previous results.
        Handles cases where explanation is a dict instead of a string.
        """
        description = []
        for path, _, output in results_list:
            description.append(f"Path: {str(path)}\nOutput{str(output)}\n\n")
        return description


    def _call_load(self):
        system_prompt = (
            """
            You are a python coding assistant with expertise in working with the Pandas library.
            Your task is to load the correct files from the list of files provided based on the task.
            
            < Task >
            {task}

            < Files available and descriptions >
            {output_descriptions}

            < Instructions >
            1. Load the csv necessary for executing python code to complete the task.
            2. Typically, you would load the output from all previous SQL programmer outputs.

            """
        )
    
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "output_descriptions"],
        )
        return prompt_template | self.load_llm    

    def _call_tool(self):
        system_prompt = (
            """
            You are a python coding assistant with expertise in working with the Pandas library.
            Your task is to transform pandas DataFrames based on user instructions.
            
            ***STRICT RULES FOR CODE GENERATION:***
            - ✅ ALWAYS return a single Python code block using triple backticks: ```python ...code... ```
            - ✅ ALWAYS begin with necessary imports, each on its own line:
                import pandas as pd
                import numpy as np (if needed)
                from scipy import ... (if needed)
            - ✅ ALWAYS work with the input DataFrames provided:
                - The first DataFrame is named 'input_df1'
                - Additional DataFrames are named 'input_df2', 'input_df3', etc.
            - ✅ ALWAYS assign the final result to a variable named `result` which MUST be a single pandas DataFrame
            - ✅ ALWAYS include comments explaining complex operations or logic
            - ❌ NEVER use file I/O operations, print statements, or display functions
            - ❌ NEVER rename columns unless explicitly requested by the user. Avoid column name conflicts when concatenating DataFrames
            - ❌ NEVER return multiple code blocks or non-code explanations mixed with code
            - ✅ If the task applies to only part of the DataFrame, return only the relevant rows or columns
            - ✅ Handle potential errors (like division by zero, missing NaN values)
            - ✅ For complex operations, use intermediate variables with descriptive names
            - ❌ NEVER perform any type of visualization

            First, think step-by-step about how to perform the task using pandas, numpy and python code.
            Then, return only the final code inside a Python code block.

            Example format of your response:
            ```python
            import pandas as pd
            import numpy as np
            
            # Your transformation code here
            df = input_df1...
            df = input_df2...
            
            # Final result assignment
            result = transformed_dataframe
            ```

            < Task >
            {task}

            < Available DataFrames >
            {dataframes_info}

            < Number of DataFrames available >
            {num_dataframes}

            After generating code, double check to make sure the columns used exist in the dataframe.
            You must respond with a tool.
            Respond only with JSON.
            """
        )
    
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "dataframes_info", "num_dataframes"],
        )
        return prompt_template | self.llm
    

    def _handle_result(self, result, import_code, python_code, explanation, output_description, df_index):
        """
        Process the result returned by executing the generated code.
        """
        working_results = []
        if isinstance(result, pd.DataFrame):
            return_file = f"{WORKING_DIRECTORY}{self.session_id}/{self.session_id}_{df_index}.csv"
            logger.info(f"\033[44m[PYTHON PROGRAMMER] Writing dataframe to {return_file}\033[0m")
            result.to_csv(return_file, index=False)
            working_results.append((return_file, explanation, output_description))
            pretty_output = pretty_print_df(result, return_output=True, max_rows=5)

            pretty_output += f"\n\nColumns in dataframe:\n{list(result.columns)}\n\nDataFrame (first few rows):\n```{result.head(5)}```\n"

        elif isinstance(result, dict):
            working_results.append((f"python_{df_index}", result, output_description))
            pretty_output = pretty_print_dict(result, return_output=True, max_items=5)
        else:
            pretty_output = str(result)

        df_index +=1
        stashed_msg = f"Agent:\nPython Programmer\n\nPython code:\n{python_code}\n\nExplanation:\n{explanation}\n\nOutput:\n{pretty_output}"

        try:
            file_path = os.path.join(WORKING_DIRECTORY, self.session_id, f"{self.session_id}_{df_index}.py")
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logger.debug(f"[PYTHON PROGRAMMER] Ensured directory exists: {os.path.dirname(file_path)}")

            with open(file_path, 'w') as file:
                # Write the imports
                file.write("# Imports\n")
                file.write(str(import_code))
                file.write("\n\n")

                # Write the explanations
                file.write("# Explanations\n")
                file.write("'''")
                file.write(str(explanation))
                file.write("'''")
                file.write("\n\n")

                # Write the Python code
                file.write("# Python Code\n")
                file.write(str(python_code))
                file.write("\n")
                
            df_index += 1
        except Exception as e:
            logger.error(f"[PYTHON PROGRAMMER] Failed to write python code to file. Error: {e}")

        return {
            "next": "QA",
            "current": "PythonProgrammer",
            "messages": [AIMessage(f"Python Programmer executed code successfully.\n\n{pretty_output}\n\n{explanation}")],
            "stashed_msg": stashed_msg,
            "working_results": working_results,
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
            "messages": [AIMessage(f"ERROR: {message}")],
            "stashed_msg": f"ERROR: {message}",
        }