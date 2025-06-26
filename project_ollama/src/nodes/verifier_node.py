import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langgraph_class.node_base import NodeBase

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
        
        elif previous_node in ['Planner']:
            logger.info(f"[VERIFIER] Routed directly from planner. Asking for human feedback first.")       
            return {"next": "HumanFeedback", "current": "Verifier", "messages": [AIMessage(f"\033[1m\033[31mAre you satisfied with the plan? If not, you may respond with changes you would like.\033[0m")]}
        
        elif previous_node in ['HumanFeedback']:
            logger.info(f"[VERIFIER] Routed from human feedback. Check if feedback is positive or negative.")       
            response = self.pos_feedback_chain.invoke({'message':last_message})
            print(response)
            return {"messages": [response], "next": "RoutingTool", "current": "Verifier"}

            # if pos_feedback_indicator == 'n':
            #     return {'next': 'Planner', "messages": [AIMessage(f"User is not satisfied with the plan. Revising...")]}
            # else:
            #     return {'next': 'RedirectTool', "messages": [AIMessage(f"User is satisfied with the plan. Executing next steps...")]}
        # elif previous_node == 'Execute':
        #     pos_feedback_indicator = self.get_pos_feedback_indicator(state)

        #     if pos_feedback_indicator == 'n':
        #         return {'next': 'Revise'}
        #     else:
        #         return {'next': 'Memorize'}
        # elif previous_node == 'Memorize':
        #     return {"messages": [AIMessage(content="Please initialize a new session for a new task")], 
        #             "next": "END"
        #         }