# InferA

## About InferA - Large-Scale Data Analysis
Inference through AI (InferA) is a data analysis multi-agent system. This project addresses the challenges of analyzing large-scale structured and numerical data using AI technologies. While LLM chatbots excel at language tasks, they struggle with massive datasets. We present a multi-agent system designed to overcome these limitations and automate complex data analysis workflows.

## Dataset

Our focus is on the HACC (Hardware/Hybrid Accelerated Cosmology Code) dataset, a terabyte-scale cosmological simulation suite. This dataset includes:
- Multiple terabytes of data
- Hundreds of timesteps
- Billions of cosmic objects
- Specialized file formats requiring expert interpretation

## Our Solution: Multi-Agent System

We've developed a multi-agent approach to break down complex data analysis tasks:
1. Planner Agent: Translates natural language queries into detailed execution plans
2. Supervisor Agent: Manages dynamic execution and task adjustments
3. Data Loading and SQL Agents: Handle staged data loading
4. Python Agent: Performs logic-intensive analysis and computation
5. Visualization Agent: Generates task-specific plots (e.g., Paraview visualizations)

![alt text](InferA/infera-workflow.png "Infera Workflow")

## Key Features
- Human-in-the-Loop: Allows continuous human feedback and supervision
- State Persistence: Saves all generated outputs and states for easy "time travel"
- Metadata-Aware Reasoning: Uses RAG for context-aware column selection
- Sandboxed Code Execution: Ensures data integrity with read-only access to the main dataset
- Optimized Token Usage: Efficient agent communication (Average run: <40,000 tokens, ~$0.09 with GPT-4)

# Getting Started


# Contact



O# (O4923)

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.




