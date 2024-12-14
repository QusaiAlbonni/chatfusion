from abc import ABC, abstractmethod
from ..prompts.prompts import BasePrompt, SingleMessagePrompt, ChatPrompt, Text, File, Part, SystemMessage

class Serializer(ABC):
    @abstractmethod
    def serialize(self, prompt: 'BasePrompt') -> any:
        pass
    
    @abstractmethod
    def get_system_instructions(self, prompt: 'BasePrompt') -> any:
        pass

class AsyncSerializer(ABC):
    @abstractmethod
    def aserialize(self, prompt: 'BasePrompt') -> any:
        pass
    