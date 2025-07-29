import os
import logging
from typing import List, TypedDict, Optional
from pydantic import Field

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase
from config import MESSAGE_HISTORY, WORKING_DIRECTORY, DISABLE_DOCUMENTATION

logger = logging.getLogger(__name__)

class DocumentationResponse(TypedDict):
    agent: str = Field(..., description="Name of the agent providing this documentation")
    task: str = Field(..., description="Description of the task being performed")
    details: str = Field(..., description="Detailed explanation of what was done, findings, and next steps")
    code: Optional[str] = Field(None, description="Any code that was generated or modified")
    challenges: Optional[str] = Field(None, description="Challenges encountered during the task")

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("Documentation")
        self.llm = llm.with_structured_output(DocumentationResponse)

        system_prompt = (
            """
            You are a meticulous research process note-taker. Your main responsibility is to observe, summarize, and document the actions and findings of the other members of your team. Your tasks include:

            1. Observing and recording key activities, decisions, and discussions among team members.
            2. Summarizing complex information into clear, concise, and accurate notes.
            3. Organizing notes in a structured format that ensures easy retrieval and reference.
            4. Highlighting significant insights, breakthroughs, challenges, or any deviations from the plan.
            5. Include entire code blocks when provided as output.
            5. Responding only in JSON format to ensure structured documentation.

            Your output should be well-organized and easy to integrate with other project documentation.

            < Entire plan >
            {plan}

            < Your last documentation >
            {last_documentation}
            """
        )
        
        self.prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            ("ai", "{stashed_msg}"),
            ("human", "{message}")
        ])

        self.chain = self.prompt_template | self.llm

    
    def run(self, state):
        if DISABLE_DOCUMENTATION:
            return {"messages": [AIMessage("Documentation skipped.")]}

        session_id = state.get("session_id", "")
        filename = f"{WORKING_DIRECTORY}{session_id}/documentation.txt"

        messages = state["messages"]
        plan = state["plan"]
        last_documentation = state.get("last_documentation", "")

        stashed_msg = state.get("stashed_msg", "")

        response = self.chain.invoke({'message': messages[-MESSAGE_HISTORY:],'plan': plan, 'stashed_msg': stashed_msg, 'last_documentation': last_documentation})
        last_documentation = self.append_documentation_to_file(response, filename, messages[0].content)

        return {"messages": [AIMessage("Documentation has been recorded.")], "last_documentation": last_documentation}
    
    
    def append_documentation_to_file(self, response, filename, message):
        file_exists = os.path.isfile(filename)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # If file does not exist, write a file header
        header = ""
        if not file_exists:
            header = "=== Documentation Log ===\n\n"
        
        # Format the content with clear section headers and spacing
        content = f"## Agent: {response.get('agent', "")}\n\n"

        content = f"## Message: {message}\n\n"
        
        # Add task and status
        content += f"**Task:** {response.get('task', "")}\n"
        
        # Add detailed description
        content += f"### Details\n{response.get('details', "")}\n\n"
        
        if response.get('code'):
            content += f"### Code\n```\n{response['code']}\n```\n\n"
        
        # Add challenges if present
        if response.get('challenges'):
            content += f"### Challenges\n{response['challenges']}\n\n"
        
        # Add separator
        content += f"{'-' * 80}\n\n"
        
        # Write to file with proper encoding
        with open(filename, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(content)
        
        return content