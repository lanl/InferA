import logging

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

logger = logging.getLogger(__name__)

class LanguageModelManager:
    def __init__(self):
        """Initialize language model manager"""
        self.logger = logger
        self.llm = None
        self.power_llm = None
        self.json_llm = None
        self.embed_llm = None
        self.token_tracker = TokenTrackingHandler()
        self.initialize_llms()

    def initialize_llms(self):
        """Initialize LLMs"""
        try:
            server = None
            model = None
            if ENABLE_LANLAI:
                server = "LanlAI Portal"
                model = LANLAI_MODEL_NAME
                self.llm = ChatOpenAI(
                    api_key = LANLAI_API_TOKEN,
                    model = LANLAI_MODEL_NAME,
                    base_url= LANLAI_API_URL,
                    http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM),
                    temperature = 0,
                    max_tokens = 4096,
                    callbacks = [self.token_tracker],
                )
                self.power_llm = ChatOpenAI(
                    api_key = LANLAI_API_TOKEN,
                    model = LANLAI_MODEL_NAME,
                    base_url= LANLAI_API_URL,
                    http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM),
                    temperature = 0.5,
                    max_tokens = 4096,
                    callbacks = [self.token_tracker],
                )
                self.json_llm = ChatOpenAI(
                    api_key = LANLAI_API_TOKEN,
                    model = LANLAI_MODEL_NAME,
                    base_url= LANLAI_API_URL,
                    http_client=DefaultHttpxClient(verify=PATH_TO_LANLCHAIN_PEM),
                    temperature = 0,
                    max_tokens = 4096,
                    model_kwargs = {"response_format": {"type": "json_object"}},
                    callbacks = [self.token_tracker],
                )
                self.embed_llm = OllamaEmbeddings(model="nomic-embed-text:latest")
                
            elif ENABLE_OPENAI:
                server = "OpenAI API"
                model = OPENAI_MODEL_NAME
                self.llm = ChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0,
                    max_tokens = 4096,
                    callbacks = [self.token_tracker],
                )
                self.power_llm = ChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0.5,
                    max_tokens = 4096,
                    callbacks = [self.token_tracker],
                )
                self.json_llm = ChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0,
                    max_tokens = 4096,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    callbacks = [self.token_tracker],
                )
                self.embed_llm = OllamaEmbeddings(model="nomic-embed-text:latest")

            else:
                server = "Local Ollama"
                model = OLLAMA_MODEL_NAME
                # Default to local Ollama model
                self.llm = ChatOpenAI(
                    model=OLLAMA_MODEL_NAME,
                    api_key = "ollama",
                    base_url = "http://localhost:11434/api/generate",
                    temperature = 0,
                    max_tokens = 4096,
                    callbacks = [self.token_tracker],
                )
                self.power_llm = ChatOpenAI(
                    model=OLLAMA_MODEL_NAME,
                    api_key = "ollama",
                    base_url = "http://localhost:11434/api/generate",
                    max_tokens = 4096,
                    temperature = 0.5,
                    callbacks = [self.token_tracker],
                )
                self.json_llm = ChatOpenAI(
                    model=OLLAMA_MODEL_NAME,
                    api_key = "ollama",
                    base_url = "http://localhost:11434/api/generate",
                    temperature = 0,
                    max_tokens = 4096,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    callbacks = [self.token_tracker],
                )
                self.embed_llm = OllamaEmbeddings(model="nomic-embed-text:latest")

            
            self.logger.info("[LLM MANAGER] LLMs initialized successfully.")
            print(f"""
            ###########################
            [Client Initialized]
            Client:      {server}
            Model:       {model}
            ###########################
            """)

        except Exception as e:
            self.logger.error(f"Error initializing language models: {str(e)}")
            raise
    
    
    def get_models(self):
        """Return all initialized language models"""
        return {
            "llm": self.llm,
            "power_llm": self.power_llm,
            "json_llm": self.json_llm
        }


class TokenTrackingHandler(BaseCallbackHandler):
    def __init__(self):
        self.logger = logger
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
            self.logger.info(f"[CURRENT USAGE]{self.get_usage()}")

    def get_usage(self):
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'cache_tokens': self.cache_tokens
        }
