import logging
import duckdb

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.utilities.sql_database import SQLDatabase

from src.langgraph_class.node_base import NodeBase
from src.utils.dataframe_utils import pretty_print_df

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, db_writer, sql_tools):
        super().__init__("SQL")
        self.llm_load = llm.bind_tools(db_writer)
        self.llm_sql = llm

        self.db_write_chain = self.load_to_db()
        self.generate_sql_chain = self.generate_sql()
    
    def run(self, state):
        # Based on user feedback, revise plan or continue to steps       
        task = state["task"]
        last_message = state["messages"][-1]

        file_index = state.get("file_index", None)
        db_path = state.get("db_path", None)
        # print("###############")
        # db_path = "./data_storage/124.duckdb"
        # print("./data_storage/124.duckdb")

        retrieved_docs = state.get("retrieved_docs", None)

        if not retrieved_docs:
            logger.info(f"[SQL PROGRAMMER] Column names not retrieved yet. Routing to retriever node.")       
            return {"next": "Retriever", "current": "SQLProgrammer", "messages": [AIMessage("Retrieving relevant columns. Sending to retriever node.")]}
        
        elif not db_path:
            response = self.db_write_chain.invoke({"task": task, "context": retrieved_docs})
            return {"next": "DBWriter", "current": "SQLProgrammer", "messages": [response]}
        
        else:
            response = self.generate_sql_chain.invoke({"task": task, "last_message": last_message, "context": retrieved_docs})
            db = duckdb.connect(db_path)
            sql_response = db.sql(response['sql']).df()
            print("SQL Filtered data:")
            pretty_print_df(sql_response)
            db.close()
            return {"next": "Supervisor", "current": "SQLProgrammer", "messages": [AIMessage(f"SQL filtered data: {sql_response.info()}")]}


    def load_to_db(self):
        load_db_prompt = (
            "You are given access to the load_to_db tool to write a database from a special file format, genericio, by extracting data from specific columns.\n\n"
            "Task:\n"
            "{task}\n\n"
            "You are also given the following context listing all column names in the data. Use this context to select columns.\n"
            "Context:\n"
            "{context}\n\n"
            "IMPORTANT:\n"
            "- Only output the exact column names to extract, separated by commas or newlines.\n"
            "- DO NOT include any explanations, extra text, or formatting.\n"
            "- Only include column names that are relevant and exactly match those in the context.\n"
            "- Do NOT output anything else besides the column names.\n"
            "- Run the load_to_db tool once with the list of selected columns.\n"
        )
        prompt_template = PromptTemplate(
            template = load_db_prompt,
            input_variables=["task", "context"]
        )
        return prompt_template | self.llm_load
    

    def generate_sql(self):
        sql_schema = ResponseSchema(name="sql", description="The SQL query to execute")
        output_parser = StructuredOutputParser.from_response_schemas([sql_schema])
        format_instructions = output_parser.get_format_instructions()

        system_prompt = (
            "You are an agent designed to interact with a SQL database. You are one of several team members who all specialize in different tasks." \
            "Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer." \
            "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results." \
            "Your job is to help other data analysis members not be overwhelmed by large amounts of data." \
            "" \
            "This is the full task:" \
            "{task}" \
            "" \
            "The last team member just completed the following task:"
            "{last_message}" \
            "" \
            "The simulation data is outlined as follows: "\
            "- simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps." \
            "- timestep: Each timestep is a folder containing cosmology particles from that simulated timestep." \
            "- cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies." \
            "" \
            "You can order the results by a relevant column to return the most interesting examples in the database." \
            "Never query for all the columns from a specific table, only ask for the relevant columns given the question." \
            "" \
            "You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again." \
            "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database." \
            "" \
            "You are given the following context containing column names in the data, use this as context for which columns to load." \
            "Context:" \
            "{context}"
            "" \
            "Only generate a SQL query to complete the task." \
            "{format_instructions}"
        )
        prompt_template = PromptTemplate(
            template = system_prompt,
            input_variables=["task", "last_message", "context"],
            partial_variables={"dialect": "PostgreSQL", "top_k": 5, "format_instructions": format_instructions},
        )

        return prompt_template | self.llm_sql | JsonOutputParser()
