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

        # Aggregate all explanations for previous dfs
        output_descriptions= self._aggregate_results_description(results_list)
        print(output_descriptions)
        try:
            response = self.call_load.invoke({
                "task": task,
                "output_descriptions": output_descriptions
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate tool: {e}")
        
        tool_calls = response.tool_calls
        data_paths = tool_calls[0]['args']['data_paths']
        logger.info(f"[PYTHON PROGRAMMER] To best answer the task, using data from {data_paths}")

        # Load dataframes for processing
        df = self._load_dataframe(data_paths)
        if df is None:
            error_msg = "[PYTHON PROGRAMMER] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        columns = list(df.columns)

        try:
            response = self.call_tool.invoke({
                "task": task, 
                "columns": columns,
                "df_head": df.head().to_string(),
                "df_describe": df.describe().to_string(),
                "df_types": df.dtypes.to_string()
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate tool: {e}")
        
        tool_calls = response.tool_calls
        if not tool_calls:
            logger.error(f"No tools called.")
            return self._error_response(f"No tools called for visualization. Must return at least one tool.")
        
        logger.debug(f"[PYTHON PROGRAMMER] Tools called: {tool_calls}")

        code_response = [t for t in tool_calls if t.get("name") == 'GenerateCode']
        other_tools = [t for t in tool_calls if t.get("name") != 'GenerateCode']

        if other_tools:
            return {"messages": other_tools, "next": "PythonTool", "current": "PythonProgrammer"}
        if code_response:
            code_response = code_response[0]

            import_code = code_response['args']['imports']
            python_code = code_response['args']['python_code']
            explanation = code_response['args']['explanation']
            output_description = code_response['args']['output_description']

            logger.info(f"\033[1;30;47m[PYTHON PROGRAMMER] Imports:\n\n{import_code}\n\nGenerated code:\n\n{python_code}\n\nExplanation:\n{explanation}\033[0m\n\n")
            # Execute the code safely from fastAPI server
            try:
                result = query_dataframe_agent(df, python_code, imports=import_code)
                if isinstance(result, dict) and "error_type" in result and "error_message" in result:
                    error_str = f"Execution returned error: {result['error_type']}: {result['error_message']}.\nCode: {python_code}"
                    return self._error_response(error_str)
                else:
                    result = pd.DataFrame.from_dict(result)

            except Exception as e:
                logger.error(f"Execution error: {e}")
                return self._error_response(f"Failed to execute code on server: {e}. Code: {python_code}.")
        
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
            combined_df = pd.concat(dfs)
            logger.debug(f"[PYTHON PROGRAMMER] Loaded and concatenated CSV dataframes.")
            return combined_df

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
            Your task is to transform a pandas DataFrame named `input_df` based on user instructions. 
            You are given tools to complete your task. Answer with maximum one tool call.
            
            ***STRICT RULES (Always Follow These if using the GenerateCode tool):***
            - ✅ ALWAYS return a single Python code block using triple backticks: ```python ...code... ```
            - ✅ ALWAYS assign the final result to a single DataFrame named `result`. It must ALWAYS be a dataframe.
            - ❌ NEVER use file I/O, or print statements.
            - ✅ Each import should be on its own separate line.
            - ✅ Use only pandas (replacing pandas), numpy, and scipy operations. If using any of the functions in these libraries, double check to make sure the namespace is correct.
            - ✅ If the user’s task applies to only part of the DataFrame, return just the relevant rows or columns.
            - ❌ Do not rename any columns in the dataframe. When possible, keep previous column names.

            First, think step-by-step about how to perform the task using pandas, numpy and python code.
            Then, return only the final code inside a Python code block.

            < Task >
            {task}

            < Columns in dataframe >
            {columns}

            < DataFrame (first few rows) >
            ```
            {df_head}
            ```

            < DataFrame Statistical Summary (`df.describe()`) >
            ```
            {df_describe}
            ```

            < DataFrame types >
            {df_types}

            """
        )
    
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "columns", "df_head", "df_describe", "df_types"],
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

        elif isinstance(result, dict):
            working_results.append((f"python_{df_index}", result, output_description))
            pretty_output = pretty_print_dict(result, return_output=True, max_items=5)
        else:
            pretty_output = str(result)

        df_index +=1
        stashed_msg = f"Python code\n{python_code}\n\nOutput:\n{pretty_output}"

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
            "messages": [AIMessage(f"Code executed successfully.\n\n{pretty_output}")],
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