import os
from dotenv import load_dotenv

load_dotenv()

# llm_client.py - model configs
# Can choose to use LanlAI portal API or a local ollama model

# LanlAI setup
ENABLE_LANLAI = False # If true, switches model to use LanlAI API
LANLAI_API_URL = "https://aiportal-api.stage.aws.lanl.gov/v2/serve"
LANLAI_API_TOKEN = os.getenv("lanlAI_token")
# LANLAI_MODEL_NAME = "meta.llama3-70b-instruct-v1:0"
# LANLAI_MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"
LANLAI_MODEL_NAME = "anthropic.claude-3-5-sonnet-20240620-v1:0"
PATH_TO_LANLCHAIN_PEM = "lanlchain.pem"

ENABLE_OPENAI = True
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
OPENAI_MODEL_NAME = "o4-mini"

# Local Ollama setup
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

# Local embedding model setup
OLLAMA_EMBEDDING_NAME = "nomic-embed-text:latest"

# logger_config.py setup
ENABLE_LOGGING = True
ENABLE_DEBUG = False
ENABLE_CONSOLE_LOGGING = True

# graph_builder.py configs - shortcut to skip user query and directly test later nodes
TEST_WORKFLOW = False # If true, swap initial state to use one of the task states. This will skip over user input.
TEST_PANDAS = False

# node_base.py configs - node-wise debug and verbose flags to print node values
NODE_VERBOSE = True
NODE_DEBUG = False