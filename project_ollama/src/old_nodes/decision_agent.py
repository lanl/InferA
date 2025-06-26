from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.langgraph_class.node_base import NodeBase
from src.prompts.prompt_templates import decision_agent_template
from src.tools.tool_registry import TOOL_REGISTRY


class DecisionAgentNode(NodeBase):
    """
    A node for executing pandas operations based on user input.

    This class extends NodeBase and provides functionality to interact with
    pandas DataFrames using natural language queries.
    """
    def __init__(self, llm):
        """
        Initialize the DecisionAgentNode.

        Args:
            llm: The language model to use for taking user query and deciding the appropriate tools and in what order
        """
        super().__init__("DecisionAgentNode")
        self.llm = llm.bind_tools(TOOL_REGISTRY)

    def run(self, state):
        user_input = state.get("user_input")
        tools = TOOL_REGISTRY

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{decision_agent_template}"), 
            ("human", "{user_input}"), 
            ("placeholder", "{agent_scratchpad}"),
        ])

        response = (prompt | self.llm).invoke({"user_input": user_input, "decision_agent_template": decision_agent_template})
        print(response)
        return {"decision_response": response.text(), "tool_calls": response.tool_calls, "messages": response}

