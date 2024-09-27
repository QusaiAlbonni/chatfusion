from .types import Optional
import os
from .model_registry import openai, genai, models, register_openai_default_provider, register_gemini_default_provider
# Default configuration


class ChatConfig:
    @staticmethod
    def set_api_provider(provider: str):
        models.set_default_provider(provider)
        
    @staticmethod
    def set_gemini_key(key: str):
        genai.configure(api_key=key)
    
    @staticmethod
    def set_openai_key(key: str):
        openai.api_key = key
        os.environ.setdefault('OPENAI_API_KEY', key)

chat_config = ChatConfig

def configure(
    register_default_providers: bool = True,
    api_provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
):
    
    if api_provider is not None:
        chat_config.set_api_provider(api_provider)
    if gemini_api_key is not None:
        chat_config.set_gemini_key(gemini_api_key)
    if openai_api_key is not None:
        chat_config.set_openai_key(openai_api_key)
        
    gemini_model = kwargs.get('gemini_model', None)
    if gemini_model is not None:
        models.set_gemini_model(gemini_model)

    openai_model = kwargs.get('openai_model', None)
    if openai_model is not None:
        models.set_openai_model(openai_model)
        
    if register_default_providers:
        register_openai_default_provider()
        register_gemini_default_provider()
    

