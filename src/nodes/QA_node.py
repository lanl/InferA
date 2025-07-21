import logging
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.nodes.node_base import NodeBase
from src.tools.human_feedback import human_feedback

from config import MESSAGE_HISTORY, DISABLE_FEEDBACK


logger = logging.getLogger(__name__)

class FormattedReview(TypedDict):
    score: int = Field(...,
        description="A numeric grade for the output between 1 and 100. "
                    "1-20: Poor, 21-40: Fair, 41-60: Good, 61-80: Very Good, 81-100: Excellent. "
                    "Consider factors like accuracy, completeness, and relevance to the task."
    )
    original_task: str = Field(...,description="Provide the original task.")
    # user_request: str = Field(...,
    #     description="Describe specific feedback the user provided, if any. "
    #                 "Include both explicit feedback and any implicit expectations or "
    #                 "requirements inferred from the user's query."
    # )
    # critique: str = Field(..., description="Clearly identifies what is missing, incorrect, or poorly executed in the output. Structure this as: 1) Strengths, 2) Weaknesses, 3) Areas for improvement.")
    revisions: str = Field(...,description="Clearly list specific, actionable revisions necessary to improve the output. Prioritize these revisions based on their potential impact.")
    before_example: str = Field(..., description="Provide a small snippet of where the error in the code is occuring.")
    after_example: str = Field(..., description="Provide an example of changes to the before_example code snippet that would fix the output.")
    summary: str = Field(..., description="A brief overall assessment of the output quality and the most critical areas for improvement.")
    confidence: float = Field(..., description="A value between 0 and 1 indicating the AI's confidence in its review and suggestions.")

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("QA")
        self.llm = llm.with_structured_output(FormattedReview)
        self.system_prompt = """
            You are a Quality Assurance (QA) expert specialized in pandas, SQL, and scientific data workflows for cosmology simulation analysis projects.

            < ROLE >
            Evaluate the latest agent response to determine if it fully and correctly completes the assigned data analysis task based on its output and user feedback.

            < GRADING >
            1. Assign a numeric grade between 1 and 100:
            - 100: Perfect and fully complete
            - 1: Completely incorrect or missing
            - If user approves, score should be between {threshold} and 100.
            - If user does not approve, score should be between 1 and {threshold}.
            2. If agent responds only with 'SUCCESS', assign a score of 100.
            3. Evaluate based on code addressing the task, assuming {member} knows all metadata details.

            < RESPONSE FORMAT >
            1. Identify specific elements that need to be changed.
            2. Provide clear, actionable feedback that directly revises the agent's output.
            3. Use user feedback to outline changes necessary.
            4. Sometimes the error is caused by a small syntax error. If so, just provide feedback on that.

            < OUTPUT FROM PREVIOUS AGENT >
            Task assigned:
            '''
            {task}
            '''

            Agent who completed the task: {member}

            Agent's last output:
            '''
            {stashed_msg}
            '''

            User feedback:
            '''
            Approved? {approved}
            {user_feedback}
            '''

            Respond only with JSON.
            """

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    
    def run(self, state):
        previous_node = state["current"]
        task = state["task"]
        messages = state["messages"]

        stashed_msg = state.get("stashed_msg", "")
        qa_retries = state.get("qa_retries", 0)

        results_list = state.get("results_list", [])
        working_results = state.get("working_results", [])
        
        # When qa_retries matches reset_retry, reset once.
        reset_retry = 2
        max_retries = 20
        threshold = 50

        feedback, approved = human_feedback(stashed_msg)
        if DISABLE_FEEDBACK:
            approved = None
            feedback = ""

        response = self.chain.invoke({
            'message': messages[-MESSAGE_HISTORY:], 
            'task':task,
            'member': previous_node, 
            'stashed_msg': stashed_msg, 
            "threshold": threshold,
            "approved": approved,
            "user_feedback": feedback
        })

        try:
            score = int(response.get("score", 0))
            original_task = response.get("original_task", "")
            revisions = response.get("task", "")
            before_example = response.get("before_example", "")
            after_example = response.get("after_example", "")
            summary = response.get("summary", "")
            confidence = float(response.get("confidence", 0))

        except Exception as e:
            logger.error(f"Failed to parse QA response JSON: {e}")
            score = 0
            original_task = ""
            revisions = ""
            before_example = ""
            after_example = ""
            summary = ""
            confidence = 0
        
        # revised_task = f"< INITIAL TASK >\n{task}\n\n< Critiques >\n{critique}\n\n< REVISIONS NEEDED >\n{revisions}\n\n"
        revised_task = f"""
ORIGINAL TASK: {original_task}

SUMMARY: {summary}

OVERALL SCORE: {score}

REVISIONS (Priority order):
{revisions}

EXAMPLE OF CHANGES:
    BEFORE: {before_example}
    AFTER: {after_example}
"""
        logger.debug(revised_task)

        if score < threshold:
            logger.info(f"Overall score: {score}\n\nSummary of changes:\n{summary}\n\nRevisions:\n{revisions}\n\n")
            qa_retries += 1
            if qa_retries > max_retries:
                return {
                    "messages": [{"role": "assistant", "content": f"❌ \033[1;31mMaximum QA retries reached with score {score}. Failed to complete the task: {task}\n\n. Escalating to Supervisor.\033[0m"}],
                    "task": revised_task,
                    "next": "Supervisor",
                    "qa_retries": 0,
                }
            if qa_retries%reset_retry==0:
                return {
                    "messages": [{"role": "assistant", "content": f"⚠️ \033[1;31mOutput from {previous_node} failed with a score of {score}. Routing back to {previous_node}. Resetting task for retries."}], 
                    "task": task,
                    "next": previous_node,
                    "qa_retries" : qa_retries
                }
            return {
                "messages": [{"role": "assistant", "content": f"⚠️ \033[1;31mOutput from {previous_node} failed with a score of {score}. Routing back to {previous_node} with updated task.\033[0m"}], 
                "task": revised_task, 
                "next": previous_node,
                "qa_retries" : qa_retries
            }
        else:
            results_list.extend(working_results)
            return {
                "messages": [{"role": "assistant", "content": f"✅ \033[1;32mOutput from {previous_node} passed with a score of {score}. Routing to Supervisor to begin next task.\033[0m"}], 
                "next": "Supervisor",
                "qa_retries": 0,
                "results_list": results_list,
                "working_results": []
            }