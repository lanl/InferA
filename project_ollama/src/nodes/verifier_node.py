import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase

logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm, tools):
        super().__init__("Verifier")
        self.llm = llm.bind_tools(tools)
        self.pos_feedback_system_prompt = (
            "If the user's message can be understood as positive affirmation with no additional information or clarifications, redirect to Supervisor Agent."
            "If there is any additional information in the user's message beyond a single positive affirmation, redirect to Planner agent to revise." \
            "" \
            "Examples of positive affirmation may include but are not limited to 'yes', 'yep', 'looks good', 'great', 'perfect'" \
            "" \
            "Redirect ONLY to either 'Planner' or 'Supervisor'. You may leave the task empty." \
            "Respond with one sentence."
        )
        self.pos_feedback_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.pos_feedback_system_prompt),
            ("user", "{message}")
        ])

        self.pos_feedback_chain = self.pos_feedback_prompt_template | self.llm
        
    
    def run(self, state):
        # Based on user feedback, revise plan or continue to steps
        last_message = [state['messages'][-1]]
        previous_node = state['current']

        if previous_node == None:
            return {'next': 'Planner'}
        
        elif previous_node in ["Planner", "SQLProgrammer", "PythonProgrammer"]:
            logger.info(f"[VERIFIER] Routed directly from planner. Asking for human feedback first.")       
            return {"next": "HumanFeedback", "current": "Verifier", "messages": [AIMessage(f"\033[1;35mAre you satisfied with the plan? If not, you may respond with changes you would like.\033[0m")]}
        
        elif previous_node in ['HumanFeedback']:
            logger.info(f"[VERIFIER] Routed from human feedback. Check if feedback is positive or negative.")       
            response = self.pos_feedback_chain.invoke({'message':last_message})
            return {"messages": [response], "next": "RoutingTool", "current": "Verifier"}