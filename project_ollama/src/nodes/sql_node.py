import logging
import duckdb

from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser

from src.nodes.node_base import NodeBase

from src.utils.dataframe_utils import pretty_print_df
from src.utils.config import WORKING_DIRECTORY

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("SQL")
        self.llm_sql = llm
        self.generate_sql = self._generate_sql()
    
    def run(self, state):
        # Based on user feedback, revise plan or continue to steps       
        task = state["task"]
        session_id = state.get("session_id", "")
        state_key = state.get("state_key", "")
        
        object_type = state.get("object_type", None)

        db_path = state.get("db_path", None)
        db_columns = state.get("db_columns", None)
        db_tables = state.get("db_tables", None)

        results_list = state.get("results_list", [])
        df_index = state.get("df_index", 0)

        if not db_path:
            logger.info(f"[SQL PROGRAMMER] Database has not been written. Routing back to supervisor.")       
            return {"next": "Supervisor", 
                    "current": "SQLProgrammer", 
                    "messages": [AIMessage("Database is missing. Check with DataLoader to verify.")]
                }
        if not db_tables:
            logger.info(f"[SQL PROGRAMMER] Database has no tables. Routing back to supervisor.")       
            return {"next": "Supervisor", 
                    "current": "SQLProgrammer", 
                    "messages": [AIMessage("Database has no tables. Check with DataLoader to verify.")]
                }
        
        try:
            table_descriptions = ""
            for i, table_name in enumerate(db_tables):
                table_descriptions += f"Table: {table_name}\nColumns in {table_name}: {', '.join(db_columns[i])}\n\n"

            logger.info(f"[SQL PROGRAMMER] SQL Query inputs:\n   - TASK: {task}\n\n{table_descriptions}")
            response = self.generate_sql.invoke({"task": task, "table_descriptions": table_descriptions})
            sql_query = response['sql']
            explanation = response['explanation']
            logger.info(f"[SQL PROGRAMMER] Generated SQL: {sql_query}\n\n{explanation}")

            # Execute SQL query
            db = duckdb.connect(db_path)
            sql_response = db.sql(sql_query).df()
            db.close()

            if sql_response.empty:
                warning_msg = "SQL executed successfully but returned no rows. The query may be too restrictive or mismatched."
                logger.warning(warning_msg)
                diagnostic_msg = f"Columns: {db_columns}\n\n{warning_msg}\n\nSQL: {sql_query}"
            else:
                print("SQL Filtered data:")
                pp_df = pretty_print_df(sql_response, return_output = True, max_rows = 5)
                
                csv_output = f"{WORKING_DIRECTORY}{state_key}/{session_id}_{df_index}.csv"
                logger.info(f"\033[44m[SQL PROGRAMMER] Writing dataframe result to {csv_output}.\033[0m")
                sql_response.to_csv(csv_output, index= False)

                df_index += 1
                results_list.append((csv_output, explanation))
                try:
                    with open(f"{WORKING_DIRECTORY}{state_key}/{session_id}_{df_index}.sql") as file:
                        # Write the explanations
                        file.write("# Explanations\n")
                        file.write(explanation)
                        file.write("\n\n")

                        # Write the SQL query
                        file.write("# SQL Query\n")
                        file.write(sql_query)
                        file.write("\n")
                    df_index += 1
                except Exception as e:
                    logger.error(f"[SQL PROGRAMMER] Failed to write SQL query to file. Error: {e}")
                return {
                    "next": "QA",
                    "current": "SQLProgrammer",
                    "messages": [AIMessage(f"{pp_df}\n\nSQL query generated.\n{explanation}")],
                    "stashed_msg": f"Columns: {db_columns}\n\nSQL query:\n{sql_query}\nExplanation:\n{explanation}",
                    "results_list": results_list,
                    "df_index": df_index
                }
            # If warning case, still send to QA with diagnostics
            return {
                "next": "QA",
                "current": "SQLProgrammer",
                "messages": [AIMessage(diagnostic_msg)],
                "stashed_msg": diagnostic_msg
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
        sql_schema = [
            ResponseSchema(
                name="sql", 
                description="The SQL query to execute"
            ),
            ResponseSchema(
                name="explanation", 
                description="Brief explanation of what the code is doing."
            ),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(sql_schema)

        system_prompt = (
            "You are a SQL generation agent for a cosmology simulation analysis pipeline. "
            "Your job is to help filter large datasets into smaller, relevant subsets for downstream analysis. "
            "You write SQL queries that extract only the essential data from the 'data' table, based on the task below.\n\n"

            "Domain Knowledge:\n"
            "- The simulation data includes: simulations (different initial conditions), timesteps (hundreds per simulation), and files for various cosmology objects.\n"
            "- Object files include: dark matter halos, halo particles, galaxies, and galaxy particles, with coordinate and physical properties.\n\n"
            "- Always include a column containing unique identifiers."
            "Task:"
            "{task}"
            ""
            "{table_descriptions}"
            ""
            "Instructions:\n"
            "- The table names and columns for each table are given above.\n"
            "- Try your best to join tables using matching unique identifier columns - usually fof_halo_tag or gal_tag."
            "- Select only the columns that are relevant to the task.\n"
            "- Never use SELECT * — be explicit about which columns to return.\n"
            "- Optionally ORDER BY a meaningful column (e.g., mass, velocity) to get significant examples.\n"
            "- NEVER make data modifications (no INSERT, UPDATE, DELETE, DROP, etc.).\n"
            "- Do not rename the columns."
            "- Always ensure your SQL is valid {dialect} syntax.\n"
            "- Only generate a SQL query — do not explain, comment, or return anything else.\n\n"
            "- If the task requires it, you may provide multiple independent SQL queries using a semi-colon as a separator."
            "- Table name is ALWAYS the same as object_type. Make sure table = object_type."
            ""
            "{format_instructions}"
            ""
            "Respond only with JSON."
        )

        prompt_template = PromptTemplate(
            template = system_prompt,
            input_variables=["task", "table_descriptions"],
            partial_variables={"dialect": "PostgreSQL", "format_instructions": output_parser.get_format_instructions()},
        )

        return prompt_template | self.llm_sql | output_parser
