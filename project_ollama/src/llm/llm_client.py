from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.callbacks.base import BaseCallbackHandler

from openai import DefaultHttpxClient

# import all lanlAI configs
from src.utils.config import (
    ENABLE_LANLAI, 
    LANLAI_API_URL, 
    LANLAI_API_TOKEN, 
    LANLAI_MODEL_NAME, 
    PATH_TO_LANLCHAIN_PEM
)
from src.utils.config import (
    ENABLE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME
)
# import all local ollama configs
from src.utils.config import (
    OLLAMA_API_URL, 
    OLLAMA_MODEL_NAME
)

from src.utils.logger_config import get_logger

logger = get_logger(__name__)

class TokenTrackingHandler(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cache_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
            self.cache_tokens += usage.get('cache_creation_input_tokens', 0) + usage.get('cache_read_input_tokens', 0)
            logger.info(f"[CURRENT USAGE]{self.get_usage()}")

    def get_usage(self):
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'cache_tokens': self.cache_tokens
        }


temperature = 0
token_tracker = TokenTrackingHandler()
if ENABLE_LANLAI:
    try:
        llm = ChatOpenAI(
            api_key = LANLAI_API_TOKEN,
            model = LANLAI_MODEL_NAME,
            base_url= LANLAI_API_URL,
            http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM),
            temperature = temperature,
            callbacks = [token_tracker]
        )
        print(f"""
            ###########################
            [Client Initialized]
            Client:      LanlAI (Uses OpenAI API spec)
            Model:       {LANLAI_MODEL_NAME}
            Temperature: {temperature}
            ###########################
        """)
        # Here temporarily, it does not exist on LanlAI
        embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    except Exception as e:
        logger.error(f"{e}")
        raise Exception(f"Failed to initialize models: {e}")
    
elif ENABLE_OPENAI:
    try: 
        llm = ChatOpenAI(
            api_key = OPENAI_API_KEY,
            model_name = OPENAI_MODEL_NAME,
        )
        embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        print(f"""
            ###########################
            [Client Initialized]
            Client:      OpenAI
            Model:       {OPENAI_MODEL_NAME}
            Temperature: Cannot use temperature as param
            ###########################
        """)
    except Exception as e:
        logger.error(f"{e}")
        raise Exception(f"Failed to initialize models: {e}")

elif ENABLE_OPENAI:
    try: 
        llm = ChatOpenAI(
            api_key = OPENAI_API_KEY,
            model_name = OPENAI_MODEL_NAME,
            callbacks = [token_tracker]
        )
        embedding_model = OpenAIEmbeddings(api_key = OPENAI_API_KEY, model="text-embedding-3-small")
        print(f"""
            ###########################
            [Client Initialized]
            Client:      OpenAI
            Model:       {OPENAI_MODEL_NAME}
            Temperature: Cannot use temperature as param
            ###########################
        """)
    except Exception as e:
        logger.error(f"{e}")
        raise Exception(f"Failed to initialize models: {e}")

else:   
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature = temperature,
            callbacks = [token_tracker]
        )
        embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        print(f"""
            ###########################
            [Client Initialized]
            Client:      Local Ollama
            Model:       {OLLAMA_MODEL_NAME}
            Temperature: {0}
            ###########################
        """)
    except Exception as e:
        logger.error(f"{e}")
        raise Exception(f"Failed to initialize models: {e}")
