import sys
import logging
from pydantic import PrivateAttr

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.callbacks.base import BaseCallbackHandler
from openai import DefaultHttpxClient

# import all lanlAI configs
from config import (
    MAX_TOKEN_LIMIT
)

from config import (
    ENABLE_LANLAI, 
    LANLAI_API_URL, 
    LANLAI_API_TOKEN, 
    LANLAI_MODEL_NAME, 
    PATH_TO_LANLCHAIN_PEM
)
from config import (
    ENABLE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    OPENAI_EMBED_MODEL_NAME
)
# import all local ollama configs
from config import (
    OLLAMA_API_URL, 
    OLLAMA_MODEL_NAME,
    OLLAMA_EMBED_MODEL_NAME
)

logger = logging.getLogger(__name__)

class LanguageModelManager:
    def __init__(self):
        """Initialize language model manager"""
        self.logger = logger
        self.server = None
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
                embed_model = OLLAMA_EMBED_MODEL_NAME + " (via Local Ollama)"
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
                self.embed_llm = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL_NAME)

            elif ENABLE_OPENAI:
                server = "OpenAI API"
                model = OPENAI_MODEL_NAME
                embed_model = OPENAI_EMBED_MODEL_NAME
                self.llm = TokenAwareChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0,
                    max_tokens = 4096,
                    # callbacks = [self.token_tracker],
                    token_tracker = self.token_tracker
                )
                self.power_llm = TokenAwareChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0.5,
                    max_tokens = 4096,
                    # callbacks = [self.token_tracker],
                    token_tracker = self.token_tracker
                )
                self.json_llm = TokenAwareChatOpenAI(
                    api_key = OPENAI_API_KEY,
                    model_name = OPENAI_MODEL_NAME,
                    temperature = 0,
                    max_tokens = 4096,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    # callbacks = [self.token_tracker],
                    token_tracker = self.token_tracker
                )
                self.embed_llm = OpenAIEmbeddings(
                    api_key = OPENAI_API_KEY,
                    model = OPENAI_EMBED_MODEL_NAME
                )

            else:
                server = "Local Ollama"
                model = OLLAMA_MODEL_NAME
                embed_model = OLLAMA_EMBED_MODEL_NAME
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
                self.embed_llm = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL_NAME)
            
            self.logger.info("[LLM MANAGER] LLMs initialized successfully.")
            self.logger.info(f"""
            ###########################
            [Client Initialized]
            Client:         {server}
            Model:          {model}
            Embed model:    {embed_model}
            ###########################
            """)
            self.server = server

        except Exception as e:
            self.logger.error(f"Error initializing language models: {str(e)}")
            raise
    
    
    def get_models(self):
        """Return all initialized language models"""
        return {
            "server": self.server,
            "llm": self.llm,
            "power_llm": self.power_llm,
            "json_llm": self.json_llm,
            "embed_llm": self.embed_llm
        }


class TokenAwareChatOpenAI(ChatOpenAI):
    _token_tracker: BaseCallbackHandler = PrivateAttr()

    def __init__(self, *args, token_tracker: BaseCallbackHandler, **kwargs):
        super().__init__(*args, callbacks=[token_tracker], **kwargs)
        self._token_tracker = token_tracker

    def invoke(self, *args, **kwargs):
        if self._token_tracker.check_limit_exceeded():
            raise RuntimeError("Token usage limit exceeded. Aborting LLM call.")
        return super().invoke(*args, **kwargs)


class TokenTrackingHandler(BaseCallbackHandler):
    def __init__(self):
        self.logger = logger
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cache_tokens = 0
        self.max_token_limit = MAX_TOKEN_LIMIT

        # Pricing per million tokens ($)
        self.cost_per_million_prompt = 2
        self.cost_per_million_completion = 8
        self.cost_per_million_cache = 0.5
    
    def on_llm_start(self, *args, **kwargs):
        if self.check_limit_exceeded():
            print(RuntimeError(f"Token usage limit exceeded: {self.max_token_limit}. Blocking LLM call."))
            logger.error(RuntimeError(f"Token usage limit exceeded: {self.max_token_limit}. Blocking LLM call."))
            sys.exit()

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
            self.cache_tokens += usage.get('cache_creation_input_tokens', 0) + usage.get('cache_read_input_tokens', 0)
            self.logger.info(f"[CURRENT USAGE]{self.get_usage()}, {self.get_cost():.6f} $")

    def check_limit_exceeded(self) -> bool:
        return self.max_token_limit is not None and self.total_tokens >= self.max_token_limit

    def get_usage(self):
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'cache_tokens': self.cache_tokens
        }
    
    def get_cost(self) -> float:
        """
        Returns the approximate cost in dollars based on token usage.
        """
        prompt_cost = self.prompt_tokens * self.cost_per_million_prompt / 1_000_000
        completion_cost = self.completion_tokens * self.cost_per_million_completion / 1_000_000
        cache_cost = self.cache_tokens * self.cost_per_million_cache / 1_000_000

        total_cost = prompt_cost + completion_cost + cache_cost
        return total_cost