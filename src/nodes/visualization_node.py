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
    def __init__(self, llm, code_tools):
        super().__init__("Visualization")
        self.llm = llm.bind_tools(code_tools)
        self.call_tools = self._call_tools()
        self.session_id = None
    
    def run(self, state):
        task = state["task"]
        self.session_id = state.get("session_id", "")

        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        output_description = self._get_last_description(results_list)

        # Load dataframes for processing
        df = self._load_last_df(results_list)
        if df is None:
            error_msg = "[VISUALIZATION] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        columns = list(df.columns)
        try:
            response = self.call_tools.invoke({
                "session_id": self.session_id,
                "task": task, 
                "output_description": output_description,
                "columns": columns,
                "df_head": df.head().to_string(),
                "df_describe": df.describe().to_string(),
                "df_types": df.dtypes.to_string()
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate code: {e}")

        tool_calls = response.tool_calls[0]
        if not tool_calls:
            logger.error(f"No tools called.")
            return self._error_response(f"No tools called. Must return at least one tool.")
    
        logger.debug(f"[VISUALIZATION] Tools called: {tool_calls}")
        
        if tool_calls.get("name") != "GenerateVisualization":
            return {"messages": [response], "next": "VisualTool", "current": "Visualization"}
        if tool_calls.get("name") == "GenerateVisualization":
            code_response = tool_calls
            
            import_code = code_response['args']['imports']
            python_code = code_response['args']['python_code']
            explanation = code_response['args']['explanation']
            output_description = code_response['args']['output_description']

            logger.debug(f"\033[1;30;47m[VISUALIZATION] Imports:\n\n{import_code}\n\nGenerated code:\n\n{python_code}\n\nExplanation:\n{explanation}\033[0m\n\n")
            # Execute the code safely from fastAPI server
            try:
                result = query_dataframe_agent([df], python_code, imports=import_code)
                if isinstance(result, pd.DataFrame):
                    pass
                elif isinstance(result, str):
                    logger.info(f"Result: {result}")
                elif isinstance(result, None):
                    logger.debug("Code completed.")
                else:
                    logger.debug(f"Result: {result}")
            except Exception as e:
                return self._error_response(f"Failed to execute code on server\n\nError:{e}\n\nCode: {import_code}\n{python_code}")
        
            return self._handle_result(result, import_code, python_code, explanation, output_description, df_index)


    def _load_last_df(self, results_list):
        """
        Load dataframe either from CSV files in results_list or from a database.
        """
        try:
            last_path, _, _ = results_list[-1]
            if last_path.endswith(".csv"):
                df = pd.read_csv(last_path)
                logger.info(f"[VISUALIZATION] Loaded dataframe from: {last_path}")
                return df
            else:
                logger.error(f"[VISUALIZATION] Last item is not a CSV: {last_path} from {results_list}")
                return None

        except Exception as e:
            logger.error(f"[VISUALIZATION] Failed to load CSV from {results_list}: {str(e)}")
            return None
    

    def _get_last_description(self, results_list):
        """
        Return only the last explanation from the results
        """
        if not results_list:
            return ""
        _, _, output_description = results_list[-1]
        return str(output_description)
    

    def _call_tools(self):
        system_prompt = (
            """
            You are a Python coding assistant specializing in scientific visualization using PyVista, VTK, and MatPlotLib.
            Your task is to generate code based on a given pandas.DataFrame named `input_df1`.
            
            < Data Visualization Rules >
            - Use PyVista/VTK for 3D data or when explicitly instructed to plot points/geometries
            - Use Matplotlib for 2D plots, time series without 3D coordinates, or statistical visualizations
            - If uncertain, default to Matplotlib for simplicity.
            - Only perform one type of visualization (e.g. only line plots, only histograms, only 3D pv data).

            < Libraries and Imports >
            import pandas as pd
            import numpy as np
            import pyvista as pv
            import matplotlib.pyplot as plt
            import vtk

            < File Output Rules >
            - Assign the output file name to a variable named 'result'
            - Write all output files to the "data_storage/{session_id}" directory.
            - Use appropriate file extensions: .vtk or .vtp for PyVista, .png for Matplotlib.

            < PyVista/VTK Visualization Guidelines >
            - Convert DataFrame columns to numpy arrays for PyVista input.
            - Use pv.PolyData for point-based data, pv.UnstructuredGrid for volumetric data.
            - Include all relevant scalar/vector fileds from the DataFrame.
            - For time series data and .pvd, use the generate_pvd_file tool given.

            < Matplotlib Visualization Guidelines >
            - Choose appropriate plot types based on data (e.g, line, scatter, bar, histogram).
            - Always include labels, titles and legends where applicable.
            - Use plt.tight_layout() to prevent overlapping elements.

            < Code Structure and Comments >
            - You will begin with 'input_df1' already loaded. Do not load any files even if the task says so.
            - Begin with a brief comment explaining the visualization purpose.
            - Group related operations with clear, descriptive comments.
            - Use meaningful variable names that reflect their content.

            < Error Handling >
            - Include basic error checking (e.g., verify required columns exist).
            - Use try-except blocks for potential issues like division by zero or invalid data types.

            Examples:

            
            Example 1. 3D Point Cloud with Scalar Field:
            ```python
            # Visualize 3D point cloud with temperature data
            points = input_df1[['x', 'y', 'z']].to_numpy()
            temp = input_df1['temperature'].to_numpy()

            # Create PolyData and add temperature as scalar field
            pdata = pv.PolyData(points)
            pdata['Temperature'] = temp

            # Create RGB colors for each point (default: blue)
            colors = np.tile([0, 0, 255], (len(points), 1))  # blue in RGB

            # Highlight the first point with red
            colors[0] = [255, 0, 0]  # red in RGB

            # Add the RGB color array to the point cloud
            pdata['RGB'] = colors  # this will be saved in the VTK

            # Save to VTK file
            result = "data_storage/temperature_point_cloud.vtk"
            pdata.save(result)

            
            Example 2. Time Series Line Plot
            # Create time series plot of temperature
            plt.figure(figsize=(10, 6))
            plt.plot(input_df1['timestamp'], input_df1['temperature'], label='Temperature')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            plt.title('Temperature Over Time')
            plt.legend()
            plt.tight_layout()

            # Save plot
            result = "data_storage/temperature_time_series.png"
            plt.savefig(result)
            plt.close()

            
            Example 3. 3D Time Series
            Call generate_pvd_file tool

            
            < Task >
            {task}

            < Output from previous step >
            {output_description}

            < Columns in input_df1 >
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

            Analyze the DataFrame to determine appropriate visualization method and columns to use. 
            Then, generate a Python code block that creates the visualization and saves it to a file. 
            Ensure the code follows the guidelines and examples provided above.

            You must call a tool.
            Respond only with JSON.
            """
        )

        
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["session_id", "task", "output_description", "columns", "df_head", "df_describe", "df_types"],
        )
        return prompt_template | self.llm
    

    def _handle_result(self, result, import_code, python_code, explanation, output_description, df_index):
        """
        Process the result returned by executing the generated pandas code.
        """
        working_results = []        
        if isinstance(result, pd.DataFrame):
            if result.empty:
                logger.info(f"[VISUALIZATION] No dataframe returned.")
            else:
                return_file = f"{WORKING_DIRECTORY}{self.session_id}/{self.session_id}_{df_index}.csv"
                logger.info(f"\033[44m[VISUALIZATION] Writing dataframe to {return_file}\033[0m")
                result.to_csv(return_file, index=False)
                df_index += 1
                working_results.append((return_file, explanation, output_description))
                pretty_output = pretty_print_df(result, return_output=True, max_rows=5)

        elif isinstance(result, dict):
            df_index += 1
            working_results.append((f"python_{df_index}", result, output_description))
            pretty_output = pretty_print_dict(result, return_output=True, max_items=5)
        else:
            pretty_output = str(result)

        stashed_msg = f"Python code\n{python_code}\n\nOutput:\n{pretty_output}\n\nIf output is a file path, then the query was successful."
        try:
            file_path = os.path.join(WORKING_DIRECTORY, self.session_id, f"{self.session_id}_{df_index}.py")

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logger.debug(f"[VISUALIZATION] Ensured directory exists: {os.path.dirname(file_path)}")

            with open(file_path, 'w') as file:
                # Write the imports
                file.write("# Imports\n")
                file.write(import_code)
                file.write("\n\n")

                # Write the explanations
                file.write("# Explanations\n")
                file.write("'''")
                file.write(explanation)
                file.write("'''")
                file.write("\n\n")

                # Write the Python code
                file.write("# Python Code\n")
                file.write(python_code)
                file.write("\n")
            df_index += 1
        except Exception as e:
            logger.error(f"[VISUALIZATION] Failed to write python code to file. Error: {e}")

        return {
            "next": "QA",
            "current": "Visualization",
            "messages": [AIMessage(f"SUCCESS. Visualization code executed successfully.\n\n{pretty_output}")],
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
            "current": "Visualization",
            "messages": [AIMessage(f"ERROR: {message}")],
            "stashed_msg": f"ERROR: {message}",
        }