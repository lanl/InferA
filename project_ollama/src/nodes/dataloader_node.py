import logging
from langchain_core.prompts import ChatPromptTemplate

from src.langgraph_class.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("DataLoader")
        self.llm = llm.bind_tools(tools)

        self.system_prompt = (
            "You are a data loading agent for simulation data. You are a part of a team of other agents that can perform specialized data analysis task."
            "Your goal is to make sure load_file_index tool is able to run."\
            "If it is not able to run, ask the user directly for the missing parameters." \
            "You do not need to provide additional information about your task, and respond only to the user." \
            "Be brief and concise."    
            )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    
    def run(self, state):
        # session_id = state['session_id']
        # config = {"metadata": {"conversation_id": session_id}}
        task = state["messages"]
        response = self.chain.invoke({'message': task})

        if not response.tool_calls:
            logger.info(f"[DATALOADER] No tools called. Routing to human feedback for more info.")
            return {"messages": [response], "next": "HumanFeedback", "current": "DataLoader"}
        else:
            logger.info(f"[DATALOADER] Tool called.")
            return {"messages": [response], "next": "DataLoaderTool", "current": "DataLoader"}