"""
File: pandas_agent_node.py

This module defines the PandasAgentNode class, which is responsible for
interacting with pandas DataFrames and executing user queries on them.

The PandasAgentNode allows users to perform simple filtering and analysis
on DataFrames using natural language queries, which are then translated
into pandas code and executed.
"""

import pandas as pd
from typing import Tuple, List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, ConfigDict

from src.langgraph_class.node_base import NodeBase
from src.llm.fastapi_client import query_dataframe_agent
from src.prompts.prompt_templates import extract_template, pandas_agent_template

from src.utils.json_loader import extract_code_block
from src.utils.dataframe_utils import pretty_print_df, pretty_print_dict
from src.utils.logger_config import get_logger
from src.utils.config import TEST_PANDAS, ENABLE_OPENAI

logger = get_logger(__name__)


class ExtractedVariables(BaseModel):
        model_config = ConfigDict(
        extra="allow"
    )

class PandasAgentNode(NodeBase):
    """
    A node for executing pandas operations based on user input.

    This class extends NodeBase and provides functionality to interact with
    pandas DataFrames using natural language queries.
    """
    def __init__(self, llm):
        """
        Initialize the PandasAgentNode.

        Args:
            llm: The language model to use for generating pandas code.
        """
        super().__init__("PandasAgentNode")
        self.llm = llm
        self.retriever = None


    def run(self, state):
        """
        Run the PandasAgentNode, allowing user interaction with the DataFrame.

        Args:
            state: The current state containing the DataFrame to query.

        Raises:
            ValueError: If the DataFrame is not found in the state.
        """
        if TEST_PANDAS:
            # Use test data if TEST_PANDAS flag set
            df = pd.read_csv("src/data/df.csv")
            print(f"[TEST PANDAS] Flag set. Using test data\n{df}")
        else:
            # Assuming state contains the dataframe to query
            df = state.get("result")

        if df is None:
            logger.error("Dataframe not found in state.")
            raise ValueError("Dataframe not found in state.")
        
        pretty_print_df(df)
        self.retriever = state.get("retriever", None)
        logger.info(f"[RAG] Retriever initialized")

        while True:
            user_input = input("\n\033[1m\033[31mI can perform simple filtering and analysis on dataframes.\nWhat would you like to analyze? (type 'exit' to quit)\033[0m\n")
            if user_input.lower() == 'exit':
                print("Exiting PandasAgentNode.")
                break
            
            try:
                retrieved_data = self.extract_variable_names(user_input)
                result = self.run_pandas_code(user_input, df, retrieved_data)
                print(type(result))
                if isinstance(result, pd.DataFrame):
                    pretty_print_df(result)
                elif isinstance(result, Dict):
                    pretty_print_dict(result)
                else:
                    print(f"\033[1m\033[31mResult:\033[0m\n{result}")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


    def extract_variable_names(self, question: str):
        # llm_json = self.llm.bind(format="json")
        llm_json = self.llm
        parser = JsonOutputParser(pydantic_object=ExtractedVariables)
        prompt = PromptTemplate(
            template= extract_template,
            input_variables=["question", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        context = self.retriever.invoke(question)
        response = (prompt | self.llm | parser).invoke({"question": question, "context": context})
        return response


    def run_pandas_code(self, question: str, df: pd.DataFrame, retrieved_data: Dict):
        """
        Generate and execute pandas code based on a user question.

        Args:
            question (str): The user's question about the DataFrame.
            df (pd.DataFrame): The DataFrame to query.

        Returns:
            The result of executing the generated pandas code.

        Raises:
            ValueError: If the DataFrame is not found or if the generated code is unsafe.
        """
        if df is None:
            raise ValueError("Dataframe not found in state.")

        prompt = PromptTemplate(
            template = pandas_agent_template, 
            input_variables=["question", "context"]
        )

        # prompt = PromptTemplate(
        #     template = """
        #     You are a helpful assistant that receives a question about a Pandas dataframe named `df`.\n
        #     Generate a single line of Python code that operates on `df` to answer the question.\n
        #     Question: {question}
        #     \n
        #     Respond with only the pandas code (e.g. df.describe(), df[df['age'] > 30])
        #     """, 
        #     input_variables=["question"]
        # )

        logger.info(f"Columns: {retrieved_data.keys()}")
        print(f"Using the following columns for analysis: {list(retrieved_data.keys())}")

        # Generate pandas code from question using LLM
        response = (prompt | self.llm).invoke({"question": question, "context": retrieved_data})
        
        # response = (prompt | self.llm).invoke({"question": question})
        pandas_code = extract_code_block(response.content)

        # # Safety check: must start with df to prevent arbitrary code execution
        if not pandas_code.startswith("python"):
            raise ValueError("Generated code does not start with 'df', refusing to execute for safety.")
        else:
            pandas_code = pandas_code[len("python"):].strip()
            
        
        # Execute the code safely from fastAPI server
        try:
            result = query_dataframe_agent(df, pandas_code)
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return f"Error executing code: {e}"

        return result