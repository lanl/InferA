import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("SQL")
        self.llm = llm.bind_tools(tools)
        self.system_prompt = (
            "You are an agent designed to interact with a SQL database. You are one of several team members who all specialize in different tasks." \
            "Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer." \
            "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results." \
            "Your job is to help other data analysis members not be overwhelmed by large amounts of data." \
            ""
            "The simulation data is outlined as follows: "
            "- simulation: Each simulation was performed with different initial conditions. The simulation directory contains timestep calculations for hundreds of timesteps." \
            "- timestep: Each timestep is a folder containing cosmology particles from that simulated timestep." \
            "- cosmology object files: Each file contains detailed coordinate and property information about various cosmology objects like dark matter halos, halo particles which form the halos, galaxies, and galaxy particles which form the galaxies." \
            "" \
            "You can order the results by a relevant column to return the most interesting examples in the database." \
            "Never query for all the columns from a specific table, only ask for the relevant columns given the question." \
            "" \
            "You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again." \
            "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database." \
            "You should query the schema of the most relevant tables."\
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    
    def run(self, state):
        # Based on user feedback, revise plan or continue to steps       
        task = state["task"]
        response = self.chain.invoke({'message': task})

        return {"messages": [response], "next": "RoutingTool"}