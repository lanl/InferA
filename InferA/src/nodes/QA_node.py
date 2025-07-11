import logging
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.nodes.node_base import NodeBase

logger = logging.getLogger(__name__)

class FormattedReview(TypedDict):
    score: int = Field(..., description="A numeric grade to the output between 1 and 100 that reflects how well the output fulfills the task requirements.")
    critique: str = Field(..., description="Clearly identifies what is missing, incorrect, or poorly executed in the output.")
    revisions: str = Field(..., description="Clearly list revisions necessary to the output.")
    example: str = Field(..., description="Provides a brief example of what changes could be made to the output.")

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("QA")
        self.llm = llm.with_structured_output(FormattedReview)
        self.system_prompt = (
            "You are a Quality Assurance (QA) expert specialized in pandas, SQL, and scientific data workflows for cosmology simulation analysis projects.\n\n"

            "### INSTRUCTIONS:\n"
            "Your role is to thoroughly evaluate whether the latest agent response fully and correctly completes the assigned data analysis task.\n"
            "First, assign a numeric grade to the output between 1 and 100 that reflects how well the output fulfills the task requirements."
            "Assume that {member} knows all the metadata details, evaluate it only on whether the code it has used is addressing the task."
            "- 100 means perfect and fully complete,\n"
            "- 1 means completely incorrect or missing,\n"
            "- Scores above {threshold} indicate acceptable output,\n"
            "- Scores {threshold} or below indicate rejection.\n\n"
            
            "If the output scores above {threshold}, redirect to Supervisor.\n"
            "If the output scores below {threshold}, redirect back to {member} with the following as part of the task parameter:\n"

            "### RESPONSE FORMAT:\n"
            "1. Clearly identify only what is missing, incorrect, or poorly executed in the previous response.\n"
            "2. Provide a brief description that outlines the necessary revisions or improvements.\n"
            "3. Highlight any critical areas where the agent must pay special attention, such as edge cases, data handling, assumptions, or formatting.\n"
            "4. If revisions are required, please provide clear, actionable feedback that directly revises the agent's output.\n"
            "5. Include the numeric grade between 1 and 100 that reflects how well the output fulfills the task requirements.\n"
            "6. Do NOT include any explanations, commentary, or extraneous text outside the prescribed response.\n"
            "7/ If code was given to you, respond with an example of specific changes to that code that are needed.\n"

            "---\n"
            "### CONTEXT TO REVIEW:\n"
            "Task assigned:\n'''{message}'''\n\n"
            "Agent who completed the task: {member}\n\n"
            "Agent's last output:\n'''{stashed_msg}'''\n\n"
            "---"
            "Respond only with JSON."
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{message}")
        ])

        self.chain = self.prompt_template | self.llm
        
    
    def run(self, state):
        previous_node = state["current"]
        task = state["task"]

        stashed_msg = state.get("stashed_msg", "")
        qa_retries = state.get("qa_retries", 0)

        reset_retry = True
        max_retries = 3
        threshold = 50

        response = self.chain.invoke({
            'message': task, 
            'member': previous_node, 
            'stashed_msg': stashed_msg, 
            "threshold": threshold,
        })

        try:
            score = int(response.get("score", 0))
            critique = response.get("critique", "")
            revisions = response.get("task", "")
            example = response.get("example", "")

        except Exception as e:
            logger.error(f"Failed to parse QA response JSON: {e}")
            score = 0
            critique = ""
            revisions = ""
            example = ""

        if score < threshold:
            revised_task = f"**INITIAL TASK:**\n{task}\n\n**Critiques:**\n{critique}\n\n**REVISIONS NEEDED:**\n{revisions}\n\n**Example:\n{example}\n\n"
            logger.info(f"Critiques:\n{critique}\n\nRevisions:\n{revisions}\n\n")
            logger.debug(revised_task)
            qa_retries += 1
            if qa_retries > max_retries:
                return {
                    "messages": [{"role": "assistant", "content": f"❌ \033[1;31mMaximum QA retries reached with score {score}. Escalating to Supervisor.\033[0m"}],
                    "task": revised_task,
                    "next": "Supervisor",
                    "qa_retries": 0,
                    "qa_failed": True
                }

            return {
                "messages": [{"role": "assistant", "content": f"⚠️ \033[1;31mOutput from {previous_node} failed with a score of {score}. Routing back to {previous_node} with updated task:\n{revised_task}\033[0m"}], 
                "task": revised_task, 
                "next": previous_node,
                "qa_retries" : qa_retries
            }
        else:
            return {
                "messages": [{"role": "assistant", "content": f"✅ \033[1;32mOutput from {previous_node} passed with a score of {score}. Routing to Supervisor to begin next task.\033[0m"}], 
                "next": "Supervisor",
                "qa_retries": 0
            }