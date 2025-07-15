import os
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
    def __init__(self, llm, tools):
        super().__init__("Visualization")
        self.llm = llm.bind_tools(tools)
        self.call_tools = self._call_tools()
        self.session_id = None
    
    def run(self, state):
        task = state["task"]
        self.session_id = state.get("session_id", "")

        db_path = state.get("db_path", None)
        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        last_explanation = self._get_last_explanation(results_list)

        # Load dataframes for processing
        df = self._load_dataframe(results_list, db_path)
        if df is None:
            error_msg = "[VISUALIZATION] Failed to load any dataframe for processing."
            return self._error_response(error_msg)
        
        columns = list(df.columns)
        try:
            response = self.call_tools.invoke({
                "task": task, 
                "columns": columns,
                "df_head": df.head().to_string(),
                "df_describe": df.describe().to_string(),
                "explanation": last_explanation,
                "df_types": df.dtypes.to_string()
            })
        except Exception as e:
            return self._error_response(f"LLM failed to generate code: {e}")

        tool_calls = response.tool_calls
        code_response = next((t for t in tool_calls if t['name'] == 'GenerateVisualization'), None)
        
        if not code_response:
            return {"messages": [response], "next": "VisualTool", "current": "Visualization"}
        import_code = code_response['args']['imports']
        python_code = code_response['args']['python_code']
        explanation = code_response['args']['explanation']

        logger.info(f"[VISUALIZATION] Generated code:\n\n\033[1;30;47m{python_code}\n\nExplanation:\n{explanation}\033[0m\n\n")
        # Execute the code safely from fastAPI server
        try:
            result = pd.DataFrame.from_dict(query_dataframe_agent(df, python_code))
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._error_response(f"Failed to execute code on server: {e}. Code: {python_code}.")
    
        return self._handle_result(result, python_code, explanation, import_code, df_index, results_list)


    def _load_dataframe(self, results_list, db_path):
        """
        Load dataframe either from CSV files in results_list or from a database.
        """
        if not results_list:
            logger.warning(f"[VISUALIZATION] No dataframes from previous steps. Getting dataframe from db.")       
            try:
                db = duckdb.connect(db_path)
                df = db.execute("SELECT * FROM data").fetchdf()
            except Exception as e:
                logger.error(f"Database load failed: {e}")
                return None

        else:
            try:
                last_path, _ = results_list[-1]
                if last_path.endswith(".csv"):
                    df = pd.read_csv(last_path)
                    logger.info(f"[VISUALIZATION] Loaded dataframe from: {last_path}")
                    return df
                else:
                    logger.error(f"[VISUALIZATION] Last item is not a CSV: {last_path}")
                    return None

            except Exception as e:
                logger.error(f"[VISUALIZATION] Failed to load CSV: {str(e)}")
                return None
    

    def _get_last_explanation(self, results_list):
        """
        Return only the last explanation from the results
        """
        if not results_list:
            return ""
        
        _, last_explanation = results_list[-1]
        if isinstance(last_explanation, dict):
            try:
                return json.dumps(last_explanation, indent=2)
            except Exception:
                return str(last_explanation)
        else:
            return str(last_explanation)
    

    def _call_tools(self):
        system_prompt = (
            """
            You are a Python coding assistant with expertise in scientific visualization using PyVista, VTK, and MatPlotLib.
            Your task is to generate code based on a given `pandas.DataFrame` named `input_df`.
            
            If you are given coordinates or instructions to plot points, you must convert this DataFrame into a `pyvista.PolyData` object for visualization and write the result to a VTK-compatible file.
            If you are NOT given coordinates or instructions to plot points, you must plot using MatPlotLib.

            ***STRICT RULES (Always Follow These):***
            - ✅ Use ONLY the following libraries: pandas as pd, numpy as np, pyvista as pv, matplotlib.pyplot as plt, and vtk.
            - ✅ If time-series or multiple time steps are detected or implied, write `.pvd` files with per-frame `.vtp` files.
            - ✅ Always assign the output file name to a single variable named 'result'.
            - ✅ Always write the final output to directory: `"data_storage/`. Output file can be .vtk, .png, or .pvd (if timeseries coordinate data).
            - ✅ The output should include all relevant scalar/vector fields and point/mesh geometry derived from the DataFrame.
            - ✅ Visualization code MUST be tailored to the task and content of the DataFrame (e.g., 3D points, mesh cells, etc.).
            - ✅ Return a single Python code block inside triple backticks: ```python ...code... ```.
            - ❌ NEVER include print statements or file reading code.
            - ✅ Use `pyvista.PolyData(...)` for visualization, not manual VTK bindings.

            ***Basic Example for coordinate data (Follow This Format When Applicable):***
            ```python
            # Convert DataFrame with x, y, z columns into PolyData and write to a .vtp file
            points = input_df[['x', 'y', 'z']].to_numpy()
            pdata = pyvista.PolyData(points)
            pdata['temperature'] = input_df['temp'].to_numpy()
            pdata.save("data_storage/visual_output.vtp")
            ```

            ***Basic Example for non-coordinate data (Follow This Format When Applicable):***
            ```python
            # Assume input_df has columns 'x' and 'y'
            plt.figure(figsize=(8,6))
            plt.plot(input_df['x'], input_df['y'], label='Line plot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Example Plot')
            plt.legend()
            plt.savefig("data_storage/plot_output.png")
            plt.close()
            ```

            ***Multi-Timestep Example (PVD):***
            - For each timestep, write a `.vtp` file and add it to a `.pvd` index.
            - Use naming like: `data_storage/visual_output/frame_000.vtp`, ..., `data_storage/visual_output/visual_output.pvd`

            All DataFrames contain point-based fields.
            First, analyze the DataFrame to determine which columns to use for visualization.
            Then, build the appropriate PyVista visualization object and save it accordingly.

            **Task:** 
            {task}

            **Previous step in the analysis pipeline:**
            {explanation}

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

            Respond only with JSON.
            """
        )

        
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["task", "columns", "df_head", "df_describe", "explanations", "df_types"],
        )
        return prompt_template | self.llm
    

    def _handle_result(self, result, python_code, explanation, import_code, df_index, results_list):
        """
        Process the result returned by executing the generated pandas code.
        """
        if isinstance(result, dict) and "error_type" in result and "error_message" in result:
            error_str = f"Execution returned error: {result['error_type']}: {result['error_message']}.\nCode: {python_code}"
            return self._error_response(error_str)
        
        elif isinstance(result, pd.DataFrame):
            return_file = f"{WORKING_DIRECTORY}{self.session_id}/{self.session_id}_{df_index}.csv"
            logger.info(f"\033[44m[VISUALIZATION] Writing dataframe to {return_file}\033[0m")
            result.to_csv(return_file, index=False)
            df_index += 1
            results_list.append((return_file, explanation))
            pretty_output = pretty_print_df(result, return_output=True, max_rows=5)

        elif isinstance(result, dict):
            df_index += 1
            results_list.append((f"python_{df_index}", result))
            pretty_output = pretty_print_dict(result, return_output=True, max_items=5)
        else:
            pretty_output = str(result)

        stashed_msg = f"Python code\n{python_code}\n\nOutput:\n{pretty_output}"
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
            "current": "Visualization",
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