from __future__ import annotations

from .generators import ResponseGenerator
from typing import Type
from .model_registry import models, Provider, ModelRegistry
from .exceptions import ModelNotFoundException


class GeneratorFactory:
    def __init__(self, registry: ModelRegistry = models):
        self.registry = registry

    def create_generator(self, provider_name: str= None, model_name: str = None, temp: float=0.7) -> ResponseGenerator:
        generator_class = None
        
        if provider_name is not None:
            generator_class = self.get_generator_class(self.registry.get_provider(provider_name))
            if model_name is None:
                model_name = self.registry.get_provider(provider_name).default_model
        elif model_name is not None:
            try:
                generator_class = self.get_generator_class(self.get_provider(model_name))
            except AttributeError:
                raise ModelNotFoundException("Gemini model not found make sure to update the gemini models if your model is not in the default ones")
        else:
            provider = self.registry.default_provider
            model_name = provider.default_model
            generator_class = self.get_generator_class(self.registry.default_provider)
            
        if generator_class is None:
            raise ValueError('Could not Find a Response Generator for this model.')
        
        return generator_class(model_name=model_name, temperature=temp)
    
    def get_provider(self, model_name: str) -> str:
        return self.registry.get_provider_by_model_name(model_name)

    def get_generator_class(self, provider: Provider) -> Type[ResponseGenerator]:
        return provider.get_generator()

