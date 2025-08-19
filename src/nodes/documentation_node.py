"""
Module: documentation_node.py
Purpose:
    This module defines the `Node` class for handling structured documentation
    in an agentic simulation pipeline. It captures task summaries, decisions, and code
    using an LLM prompt template, formats the documentation, and saves it to a local file.

    The documentation agent ensures transparency, traceability, and reproducibility
    by keeping track of all activities in a structured JSON format.

Classes:
- DocumentationResponse: TypedDict defining the expected structured output from the LLM.
- Node(NodeBase): Agent node that generates, formats, and stores structured documentation.

Usage:
    - Automatically records task results and decisions made by agents.
    - Saves detailed documentation logs with optional code and challenges.
    - Can be disabled via `DISABLE_DOCUMENTATION` config flag.
"""

import os
import logging
from typing import TypedDict, Optional
from pydantic import Field

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node_base import NodeBase
from config import MESSAGE_HISTORY, WORKING_DIRECTORY, DISABLE_DOCUMENTATION

logger = logging.getLogger(__name__)



class DocumentationResponse(TypedDict):
    """
    Expected schema for the documentation agent's structured output.
    """
    agent: str = Field(..., description="Name of the agent providing this documentation")
    task: str = Field(..., description="Description of the task being performed")
    details: str = Field(..., description="Detailed explanation of what was done, findings, and next steps")
    code: Optional[str] = Field(None, description="Any code that was generated or modified")
    challenges: Optional[str] = Field(None, description="Challenges encountered during the task")

class Node(NodeBase):
    """
    Documentation node for summarizing and recording agent behavior.

    This node uses a system-level LLM prompt to generate structured notes
    summarizing the session. Documentation is saved to a local file named
    after the session ID.

    Attributes:
        llm: Language model instance configured to return structured JSON.
        prompt_template: A chainable chat prompt template with system, AI, and user messages.
        chain: The combined prompt and model pipeline.
    """
    def __init__(self, llm):
        """
        Initialize the documentation node with a structured-output-capable LLM.

        Args:
            llm: A language model instance (e.g. from LangChain) with structured output support.

        Customization:
            - You can modify the `system_prompt` template to change the documentation style.
        """
        super().__init__("Documentation")
        self.llm = llm.with_structured_output(DocumentationResponse)

        # Defines instructions to the LLM on how to behave as a documentation agent
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
        
        # Prompt includes system instruction, last AI output, and most recent user message
        self.prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            ("ai", "{stashed_msg}"),
            ("human", "{message}")
        ])

        self.chain = self.prompt_template | self.llm

    
    def run(self, state):
        """
        Executes the documentation process.

        - Invokes the LLM to create a structured documentation entry.
        - Appends the entry to a per-session log file.
        - Skips documentation if `DISABLE_DOCUMENTATION` is enabled.

        Args:
            state (dict): The simulation state with keys:
                - session_id (str): Unique session ID to identify the log file.
                - messages (List): Recent message history.
                - plan (str): Current simulation plan or task breakdown.
                - last_documentation (str): Previous documentation content.
                - stashed_msg (str): The last AI-generated message.

        Returns:
            dict: A state update with a new message and updated documentation log.
        """
        if DISABLE_DOCUMENTATION:
            return {"messages": [AIMessage("Documentation skipped.")]}

        session_id = state.get("session_id", "")
        filename = f"{WORKING_DIRECTORY}{session_id}/documentation.txt"

        messages = state["messages"]
        plan = state["plan"]
        last_documentation = state.get("last_documentation", "")
        stashed_msg = state.get("stashed_msg", "")

        # Run the documentation chain using recent messages and the plan
        response = self.chain.invoke({
            'message': messages[-MESSAGE_HISTORY:],
            'plan': plan, 
            'stashed_msg': stashed_msg, 
            'last_documentation': last_documentation
        })

        # Append the result to the documentation file and return updated content
        last_documentation = self.append_documentation_to_file(response, filename, messages[0].content)

        return {
            "messages": [AIMessage("Documentation has been recorded.")], 
            "last_documentation": last_documentation
        }
    
    
    def append_documentation_to_file(self, response, filename, message):
        """
        Appends a formatted documentation response to the specified file.

        - Creates the file and directory if they do not exist.
        - Includes a file header the first time the file is written.
        - Structures content with clear sections (task, details, code, challenges).

        Args:
            response (dict): The structured documentation returned by the LLM.
            filename (str): Full path to the documentation file.
            message (str): Original message that triggered the documentation step.

        Returns:
            str: The string content that was appended to the file.
        """
        file_exists = os.path.isfile(filename)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # If file does not exist, write a file header
        header = ""
        if not file_exists:
            header = "=== Documentation Log ===\n\n"
        
        # Construct formatted documentation block
        content = f"## Agent: {response.get('agent', "")}\n\n"
        content = f"## Message: {message}\n\n"
        content += f"**Task:** {response.get('task', "")}\n"
        content += f"### Details\n{response.get('details', "")}\n\n"
        
        if response.get('code'):
            content += f"### Code\n```\n{response['code']}\n```\n\n"
        
        if response.get('challenges'):
            content += f"### Challenges\n{response['challenges']}\n\n"
        
        content += f"{'-' * 80}\n\n"
        
        # Write to file with proper encoding
        with open(filename, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(content)
        
        return content