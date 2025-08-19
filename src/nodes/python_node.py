"""
Module: python_node.py

Purpose:
    This module defines a `PythonProgrammer` node that executes complex data analysis 
    tasks using Pandas and NumPy. It receives task instructions, loads relevant CSV data,
    invokes a language model to generate Python code, executes the code on the data, and
    returns the result along with explanations. It is designed for use within a multi-agent
    pipeline for cosmology simulations or other scientific analysis workflows.

Classes:
    - Node: Executes Python code generation and analysis using structured LLM tools.

Functions:
    - run(state): Main execution flow. Loads data, invokes LLM, runs generated code, and handles result.
    - _load_dataframe(data_path): Loads one or more CSV files from the given paths.
    - _aggregate_results_description(results_list): Aggregates explanations from previous steps.
    - _call_load(): Creates a prompt chain to select data files to load.
    - _call_tool(): Creates a prompt chain to generate transformation code for the loaded data.
    - _handle_result(...): Executes and logs results from generated code.
    - _error_response(message): Returns standardized error response.

Usage:
    Instantiate this node with an LLM and tool configurations:
        node = Node(llm, code_tools, load_tools)
        result = node.run(state_dict)
"""

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
    """
    PythonProgrammer Node: Executes Python-based data analysis using LLM-generated code.

    Attributes:
        llm: LLM bound with toolset for code generation.
        load_llm: LLM bound with file selection tools.
        session_id (str): Unique session ID for tracking results.
        call_load: Prompt chain for data file selection.
        call_tool: Prompt chain for code generation and transformation.
    """
    def __init__(self, llm, code_tools, load_tools):
        super().__init__("Python")
        self.load_llm = llm.bind_tools(load_tools, parallel_tool_calls = False)
        self.llm = llm.bind_tools(code_tools, parallel_tool_calls = False)
        self.call_load = self._call_load()
        self.call_tool = self._call_tool()
        self.session_id = None
    
    def run(self, state):
        """
        Executes the main logic of the node.

        Args:
            state (dict): Dictionary with keys including:
                - task (str): Analysis instruction
                - session_id (str): Session identifier
                - results_list (list): Previous tool outputs
                - df_index (int): Index for output filenames

        Returns:
            dict: Resulting state for pipeline continuation
        """
        task = state["task"]
        self.session_id = state.get("session_id", "")

        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        # Describe all previously generated results
        output_descriptions= self._aggregate_results_description(results_list)

        try:
            # LLM selects which data files to load
            response = self.call_load.invoke({
                "task": task,
                "output_descriptions": output_descriptions
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate tool.\nError:\n{e}")
                
        tool_calls = response.tool_calls
        data_paths = tool_calls[0]['args']['data_paths']
        logger.info(f"[PYTHON PROGRAMMER] Using data from {data_paths}")

        # Load the selected dataframes
        dfs = self._load_dataframe(data_paths)
        if dfs is None:
            error_msg = "[PYTHON PROGRAMMER] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        # Summarize all DataFrames for the LLM
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
            # Generate Python transformation code based on task
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
            # If another tool is called, route to tool executor
            return {"messages": [response], "next": "PythonTool", "current": "PythonProgrammer"}

        # Extract and execute code
        code_response = tool_calls
        import_code = code_response['args']['imports']
        python_code = code_response['args']['python_code']
        result_output_code = code_response['args']['result_output_code']
        explanation = code_response['args']['explanation']
        output_description = code_response['args']['output_description']
        python_code = python_code + "\n" + result_output_code

        logger.debug(
            f"\033[1;30;47m[PYTHON PROGRAMMER] Imports:\n\n{import_code}\n\n"
            f"Generated code:\n\n{python_code}\n\nExplanation:\n{explanation}\033[0m\n\n")
        

        # Execute the code safely from fastAPI server
        try:
            result = query_dataframe_agent(dfs, python_code, imports=import_code)
            # Normalize result type
            if isinstance(result, pd.Series):
                result = result.to_frame()
            elif result is None:
                result = pd.DataFrame()
            elif isinstance(result, str):
                result = pd.DataFrame({'result': [result]})
            elif not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({'result': [str(result)]})
        except Exception as e:
            return self._error_response(f"Failed to execute code on server\n\nError:{e}\n\nCode: {import_code}\n{python_code}")
    
        return self._handle_result(result, import_code, python_code, explanation, output_description, df_index)


    def _load_dataframe(self, data_path):
        """
        Load one or more CSV files as DataFrames.

        Args:
            data_path (list[str]): List of file paths

        Returns:
            list[pd.DataFrame] or None
        """
        dfs = []
        try:
            for path in data_path:
                if path.endswith(".csv"):
                    dfs.append(pd.read_csv(path))
            if not dfs:
                logger.error(f"No CSV files found in results_list: {data_path}.")
                return None
            logger.debug(f"[PYTHON PROGRAMMER] Loaded and concatenated CSV dataframes.")
            return dfs

        except Exception as e:
            logger.error(f"[PYTHON PROGRAMMER] Failed to load CSV from {data_path}: {str(e)}")
            return None
    

    def _aggregate_results_description(self, results_list):
        """
        Aggregates explanation snippets from earlier results.

        Args:
            results_list (list): List of previous tool outputs

        Returns:
            list[str]: List of formatted description strings
        """
        description = []
        for path, _, output in results_list:
            description.append(f"Path: {str(path)}\nOutput{str(output)}\n\n")
        return description


    def _call_load(self):
        """
        Builds the LLM chain for selecting which CSV files to load.
        """
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
        """
        Builds the LLM chain for generating transformation code.
        """
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
        Post-processes execution result: saves files, formats message.

        Args:
            result (pd.DataFrame|dict|str): Result from executing the generated code
            import_code (str): Import statements
            python_code (str): Executable Python code
            explanation (str): Explanation from LLM
            output_description (str): Description of result
            df_index (int): Output file index

        Returns:
            dict: State to pass to next pipeline node
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
                file.write("# Imports\n" + str(import_code) + "\n\n")
                file.write("# Explanations\n'''\n" + str(explanation) + "\n'''\n\n")
                file.write("# Python Code\n" + str(python_code) + "\n")
                
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
        Handles consistent error messaging across the node.

        Args:
            message (str): Error details

        Returns:
            dict: Error response state
        """
        logger.error(f"\033[1;31m{message}\033[0m")
        return {
            "next": "QA",
            "current": "PythonProgrammer",
            "messages": [AIMessage(f"ERROR: {message}")],
            "stashed_msg": f"ERROR: {message}",
        }