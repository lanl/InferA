import os
from dotenv import load_dotenv

load_dotenv()

# llm_client.py - model configs
# Can choose to use LanlAI portal API or a local ollama model

# LanlAI setup
ENABLE_LANLAI = True # If true, switches model to use LanlAI API
LANLAI_API_URL = "https://aiportal-api.stage.aws.lanl.gov/v2/serve"
LANLAI_API_TOKEN = os.getenv("lanlAI_token")
# LANLAI_MODEL_NAME = "meta.llama3-70b-instruct-v1:0"
# LANLAI_MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"
LANLAI_MODEL_NAME = "anthropic.claude-3-5-sonnet-20240620-v1:0"
PATH_TO_LANLCHAIN_PEM = "lanlchain.pem"

ENABLE_OPENAI = False
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
OPENAI_MODEL_NAME = "gpt-4.1-mini"
OPENAI_EMBED_MODEL_NAME = "text-embedding-3-large"

# Local Ollama setup
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

# Local embedding model setup
OLLAMA_EMBED_MODEL_NAME = "nomic-embed-text:latest"

# Working directory to save files
WORKING_DIRECTORY = "./data_storage/"
STATE_DICT_PATH = "./state/state.pkl"

# logger_config.py setup
ENABLE_LOGGING = True
ENABLE_DEBUG = False
ENABLE_CONSOLE_LOGGING = True
# ENABLE_CONSOLE_LOGGING = False

# graph_builder.py configs - shortcut to skip user query and directly test later nodes
# DISABLE_FEEDBACK = False
DISABLE_FEEDBACK = True
