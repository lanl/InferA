"""
Module: sql_programmer_node.py
Purpose: This module defines the SQL Programmer node that generates, executes,
         and logs SQL queries for simulation-based cosmology datasets.

This node connects to a DuckDB database, executes structured SQL generated
from a language model, logs outputs, and passes results downstream.

Designed to work directly after the DataLoader node and before QA evaluation.

Main responsibilities:
- Generate SQL queries using LLM
- Execute those queries against a DuckDB database
- Store output CSVs and query logs
- Provide explanation and diagnostics for downstream review
"""

import os
import logging
import duckdb

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser

from src.nodes.node_base import NodeBase
from src.utils.dataframe_utils import pretty_print_df
from config import WORKING_DIRECTORY

logger = logging.getLogger(__name__)



class Node(NodeBase):
    """
    SQL Programmer Node

    This node takes a task description and schema metadata, then:
    - Generates SQL queries via LLM prompt
    - Executes queries using DuckDB
    - Outputs results as CSV
    - Logs explanation and diagnostics
    """
    def __init__(self, llm):
        super().__init__("SQL")
        self.llm_sql = llm
        self.generate_sql = self._generate_sql()
    
    def run(self, state):
        """
        Main function that:
        - Extracts context
        - Prompts LLM to generate SQL
        - Executes SQL using DuckDB
        - Returns outputs or diagnostics

        Args:
            state (dict): Contains task, db metadata, results, session ID, etc.

        Returns:
            dict: New state with result paths, messages, or QA escalation
        """
        task = state["task"]
        session_id = state.get("session_id", "")
        
        db_path = state.get("db_path", None)
        db_columns = state.get("db_columns", None)
        db_tables = state.get("db_tables", None)

        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        # ---- Database validation ----
        if not db_path:
            logger.warning(f"[SQL PROGRAMMER] Database has not been written. Routing back to documentation/supervisor.")       
            return {
                "next": "Documentation", 
                "current": "SQLProgrammer", 
                "messages": [AIMessage("Database is missing. Check with DataLoader to verify.")]
            }
        
        if not db_tables:
            logger.warning(f"[SQL PROGRAMMER] Database has no tables. Routing back to documentation/supervisor.")       
            return {
                "next": "Documentation", 
                "current": "SQLProgrammer", 
                "messages": [AIMessage("Database has no tables. Check with DataLoader to verify.")]
            }
    
        try:
            # ---- Build schema string for LLM prompt ----
            table_descriptions = ""
            for i, table_name in enumerate(db_tables):
                table_descriptions += (
                    f"Table: {table_name}\nColumns in {table_name}: {', '.join(db_columns[i])}\n\n"
                    f"Object_type = {table_name}\n\n"
                )

            logger.info(f"[SQL PROGRAMMER] SQL Query inputs:\n\nTASK: {task}\n\n{table_descriptions}\n\n")
            
            response = self.generate_sql.invoke({
                "task": task, 
                "table_descriptions": table_descriptions
            })

            sql_query = response['sql']
            explanation = response['explanation']
            output_description = response['output_description']

            logger.info(f"[SQL PROGRAMMER] Generated SQL: \n\n\033[1;30;47m{sql_query}\n\n{explanation}\033[0m\n\n")

            # ---- Execute SQL queries ----
            db = duckdb.connect(db_path)
            all_results, all_dfs, all_csv_outputs = [], [], []

            # Split the SQL query into multiple queries if needed
            sql_queries = sql_query.split(';')
            sql_queries = [q.strip() for q in sql_queries if q.strip()]  # Remove empty queries

            for i, query in enumerate(sql_queries):                
                try:   
                    # Execute SQL query
                    sql_response = db.sql(query).df()
            
                    if sql_response.empty:
                        warning_msg = "SQL executed successfully but returned no rows. The query may be too restrictive or mismatched."
                        logger.warning(warning_msg)
                        diagnostic_msg = f"Columns: {db_columns}\n\n{warning_msg}\n\nSQL: {sql_query}"
                    else:
                        pp_df = pretty_print_df(sql_response, return_output = True, max_rows = 5)
                        all_results.append(f"Query {i+1} Results:\n{pp_df}")
                        all_dfs.append(sql_response)
                        
                        # Save CSV for this query
                        csv_output = f"{WORKING_DIRECTORY}{session_id}/{session_id}_{df_index}_query{i+1}.csv"
                        logger.info(f"\033[44m[SQL PROGRAMMER] Writing dataframe result for query {i+1} to {csv_output}.\033[0m")
                        sql_response.to_csv(csv_output, index= False)
                        all_csv_outputs.append(csv_output)

                except Exception as e:
                    error_msg = f"Error executing query {i+1}: {str(e)}\nQuery: {query}"
                    logger.error(error_msg)
                    return {
                        "next": "QA",
                        "current": "SQLProgrammer",
                        "messages": [AIMessage(error_msg)],
                        "stashed_msg": error_msg
                    }

            db.close()
            df_index += 1
            
            # ---- Save query and explanation to .sql file ----
            try:
                file_path = os.path.join(WORKING_DIRECTORY, session_id, f"{session_id}_{df_index}.sql")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                logger.debug(f"[VISUALIZATION] Ensured directory exists: {os.path.dirname(file_path)}")

                with open(file_path, 'w') as file:
                    file.write("-- Explanations\n")
                    file.write("-- " + str(explanation) + "\n\n")
                    file.write("-- SQL Query\n")
                    file.write(sql_query + "\n")

                df_index += 1

            except Exception as e:
                logger.error(f"[SQL PROGRAMMER] Failed to write SQL query to file. Error: {e}")

            # ---- Final output formatting ----
            combined_results = "\n\n".join(all_results)

            if not all_dfs:
                diagnostic_msg = f"Columns: {db_columns}\n\nNo queries returned data.\n\nSQL: {sql_query}"
                return {
                    "next": "QA",
                    "current": "SQLProgrammer",
                    "messages": [AIMessage(diagnostic_msg)],
                    "stashed_msg": diagnostic_msg
                }

            for i, csv_path in enumerate(all_csv_outputs):
                results_list.append((csv_path, explanation, f"SQL Programmer (Query {i+1}): {output_description}"))

            return {
                "next": "QA",
                "current": "SQLProgrammer",
                "messages": [AIMessage(f"{combined_results}\n\nSQL queries generated.\n{explanation}")],
                "stashed_msg": f"{combined_results}\n\nColumns: {db_columns}\n\nSQL queries:\n{sql_query}\nExplanation:\n{explanation}",
                "results_list": results_list,
                "df_index": df_index
            }

        except Exception as e:
            error_msg = f"Columns: {db_columns}\n\n[SQL ERROR] {str(e)}"
            logger.error(error_msg)
            return {
                "next": "QA",
                "current": "SQLProgrammer",
                "messages": [AIMessage(error_msg)],
                "stashed_msg": error_msg
            }
        

    def _generate_sql(self):
        """
        Builds a structured SQL generation chain using a system prompt, response schemas,
        and a structured output parser to ensure valid SQL and explanations.
        """
        sql_schema = [
            ResponseSchema(name="sql", description="The SQL query to execute"),
            ResponseSchema(name="explanation", description="Brief explanation of what the code is doing. Make sure to discuss which table the SQL query is operating on."),
            ResponseSchema(name="output_description",description="Briefly describe what the output is and what it looks like.")
        ]

        output_parser = StructuredOutputParser.from_response_schemas(sql_schema)

        system_prompt = """
        You are a SQL generation agent for a cosmology simulation analysis pipeline. 
        Your task is to write SQL queries that extract essential data from the data based on your given task, filtering large datasets into smaller, relevant subsets for downstream analysis.

        < Domain Knowledge >
        - The simulation data includes: simulations (different initial conditions as integers), timesteps (hundreds per simulation), and files for various cosmology objects.
        - Object files include: dark matter halos, halo particles, galaxies, and galaxy particles, with coordinate and physical properties.
        - Each object has its own set of 3D coordinates, physical properties, and unique identifier.

        < Task >
        {task}

        < Table names and columns in each table >
        {table_descriptions}

        < Syntax Instructions >
        1. Only use the table names and columns provided above. Check twice to make sure the columns you provide exist in the corresponding table.
        2. Ensure the table name correctly matches the above and do not make your own table ( example: table = haloproperties ).

        < Instructions >
        1. Include all columns provided in the database. Never use 'SELECT *'.
        2. Always include a column with unique identifiers.
        3. Always include a column with x, y, z coordinate data if it exists in the table.
        4. Join tables using matching unique identifier columns (usually fof_halo_tag or gal_tag)
        5. Optionally ORDER BY a meaningful column for significant examples.
        6. Use valid {dialect} SQL syntax.
        7. Do not modify data (no INSERT, UPDATE, DELETE, DROP, etc.).
        8. Do not rename the columns. Do not create a view.
        
        You may provide multiple independent SQL queries separated by semicolons if required.
        However, your job is not to do analysis, only to filter the data.

        {format_instructions}

        Respond only with JSON.
        """

        prompt_template = PromptTemplate(
            template = system_prompt,
            input_variables=["task", "table_descriptions"],
            partial_variables={"dialect": "PostgreSQL", "format_instructions": output_parser.get_format_instructions()},
        )

        return prompt_template | self.llm_sql | output_parser
