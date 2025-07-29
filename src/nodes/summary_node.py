import logging
import pandas as pd

from langchain.prompts import PromptTemplate

from src.nodes.node_base import NodeBase
from src.utils.dataframe_utils import pretty_print_df, pretty_print_dict



logger = logging.getLogger(__name__)

class Node(NodeBase):
    def __init__(self, llm):
        super().__init__("Summary")
        self.llm = llm
        self.generate_summary = self._generate_summary()
    
    def run(self, state):
        user_inputs = state.get("user_inputs", "")
        results_list = state.get("results_list", [])
        
        last_result = results_list[-1]
        path = last_result[0]
        explanation = last_result[1]
        if last_result[0].endswith(".csv"):
            df = pd.read_csv(path)
        else:
            # Not actually a df. Just the name of the result as an idx
            df = explanation

        if isinstance(df, pd.DataFrame):
            pp_df = pretty_print_df(df, return_output = True)
            summary = self.generate_summary.invoke({
                "task": user_inputs,
                "explanation": explanation,
                "df_head": df.head(), 
                "df_describe": df.describe()
            })
        else:
            pp_df = pretty_print_dict(df, return_output = True)
            summary = self.generate_summary.invoke({
                "task": user_inputs,
                "explanation": explanation,
                "df_head": "Not a dataframe.", 
                "df_describe": "Not a dataframe."
            })
        
        return {"messages": [f"{pp_df}\n\n\033[1;35m{summary.content}\033[0m"], "next": "Documentation", "current": "Summary"}
    

    def _generate_summary(self):
        system_prompt = (
            """
            You are a data analysis expert with a strong background in scientific research writing. 
            Your task is to interpret the results of a pandas-based data analysis and summarize the findings clearly and concisely, using a professional, research-style tone.

            This was the user's original task:
            {task}
            
            ---
            ### CONTEXT

            **Explanation of output from last member:**
            {explanation}

            **Resulting DataFrame (first few rows):**
            ```
            {df_head}
            ```

            **DataFrame Statistical Summary (`df.describe()`):**
            ```
            {df_describe}
            ```

            ---
            ### INSTRUCTIONS

            1. Summarize the key results — trends, relationships, anomalies, or distributions.
            2. Use technical language suitable for scientific communication.
            3. Be as descriptive as possible.
            4. Ground all claims in the data provided. Do not speculate beyond the evidence.

            ---
            ### OUTPUT

            Write a paragraph summarizing the results. Use complete sentences in a research or data-reporting tone.
            """
        )
        
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["explanation", "df_head", "df_describe", "task"]
        )
        return prompt_template | self.llm
