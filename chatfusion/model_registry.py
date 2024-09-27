from .types import Any, Dict, Optional, TYPE_CHECKING, List, Tuple
from .providers import Provider

if TYPE_CHECKING:
    from .generators import ResponseGenerator


def import_ai_libs():
    try:
        import openai
    except ImportError:
        openai = None
    try:
        import google.generativeai as genai

    except ImportError as e:
        genai = None
    return genai, openai


genai, openai = import_ai_libs()


class ModelRegistry:
    def __init__(self):
        self.providers: List[Provider] = []
        self.default_provider: Optional[Provider] = None

    def add_provider(self, provider: Provider):
        self.providers.append(provider)
        if not self.default_provider:
            self.default_provider = provider

    def get_provider(self, provider_name: str) -> Optional[Provider]:
        return next((p for p in self.providers if p.name == provider_name), None)
    
    def get_provider_by_model_name(self, model_name: str) -> Optional[Provider]:
        for provider in self.providers:
            if model_name in provider.models:
                return provider
        return None

    def set_default_provider(self, provider_name: str):
        provider = self.get_provider(provider_name)

        if provider:
            self.default_provider = provider

    def list_models(self) -> List[Tuple[str, str]]:
        return [(model, provider.name) 
                for provider in self.providers 
                for model in provider.models]

    def clear(self):
        for provider in self.providers:
            provider.models.clear()

models = ModelRegistry()






def update_gemini_models():
    models_list = genai.list_models()
    for model in models_list:
        name_without_prefix = model.name.removeprefix('models/')
        models.get_provider('gemini').set_model(name_without_prefix, {'name': model.name})


def update_openai_models():
    models_list = openai.models.list()

    for model in models_list:
        name = model.id
        models.get_provider('openai').set_model(name, {})


gemini_provider = Provider('gemini', 
                           default_model='gemini-1.5-pro-latest',
                           initial_models={

                               'gemini-1.5-pro-latest': {'some': 'data'},
                               'gemini-1.5-pro': {'other': 'data'},
                               'gemini-1.0-pro': {'other': 'data'},
                               'gemini-pro': {'other': 'data'},
                               'gemini-1.5-flash': {'other': 'data'}
                           })

openai_provider = Provider('openai', 
                           default_model='gpt-4o-mini', 
                           initial_models={
                               'gpt-4o-mini': {'some': 'data'},
                               'gpt-3.5-turbo': {'other': 'data'}
                           })

def register_openai_default_provider():
    from .generators import OpenAiGenerator
    
    models.add_provider(openai_provider)
    openai_provider.set_generator(OpenAiGenerator)
    

def register_gemini_default_provider():
    from .generators import GeminiGenerator
    
    models.add_provider(gemini_provider)
    gemini_provider.set_generator(GeminiGenerator)