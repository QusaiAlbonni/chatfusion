from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Generator, List
from .types import TYPE_CHECKING, Generator, Union
from .model_registry import genai, openai
from .exceptions import InvalidPrompt, UnexpectedBehavior, Forbidden
if TYPE_CHECKING:
    from .generators import ResponseGenerator
    from .prompts import BasePrompt


class BaseResponse(ABC):
    @abstractmethod
    def get_text(self, index=0) -> Union[str, Generator[str, None, None]]:
        pass


class Response(BaseResponse):
    def __init__(self, response, streamed: bool, generator: 'ResponseGenerator', prompt: 'BasePrompt'):
        self._response = response
        self.streamed = streamed
        self.choices = self._get_choices()
        self.generator = generator

    def _get_choices(self) -> List:
        return getattr(self._response, 'choices', [self._response])

    def get_text(self, index=0) -> Union[str, Generator[str, None, None]]:
        if self.streamed:
            return self.stream_text(index)
        return self.text(index)
    
    def get_original_response(self):
        return self._response

    def get_choice(self, index=0):
        if index < 0 or index >= len(self.choices):
            raise IndexError("Choice index out of range")
        return self.choices[index]

    def __len__(self):
        return len(self.choices)

    def text(self, index=0) -> str:
        choice = self.get_choice(index)
        if not self.is_choice_safe(index):
            reason = self.get_finish_reason(choice)
            match reason:
                case 'SAFETY':
                    raise InvalidPrompt(f"model did not finish the response properly", 
                                            f"{reason}")
                case 'FUNCTION_CALL' | 'TOOL_CALL':
                    raise Forbidden(f"calling text() on a response that asked for a function call", 
                                            f"{reason}")
                case 'MAX_TOKENS':
                    raise InvalidPrompt("Model could not parse prompt as it went over the token limit", 
                                            f"{reason}")
                case _:
                    raise UnexpectedBehavior("API returned an unexpected finish reason","UKNOWN")
        if self.streamed:
            raise Forbidden(f"calling text() on a streamed response; use stream_text")
        return self.get_choice_content(choice)

    @abstractmethod
    def stream_text(self, index=0) -> Generator[str, None, None]:
        pass
    
    @abstractmethod
    def is_choice_safe(self, index=0) -> bool:
        pass

    @abstractmethod
    def get_choice_content(self, choice):
        pass


class OpenAIResponse(Response):
    def __init__(self, response, streamed: bool, generator: 'ResponseGenerator', prompt: 'BasePrompt'):
        super().__init__(response, streamed, generator, prompt)

    def _get_choices(self) -> List:
        return self._response.choices if hasattr(self._response, 'choices') else [self._response]

    def stream_text(self, index=0) -> Generator[str, None, None]:
        for chunk in self._response:
            if chunk.choices[index].delta.content is not None:
                yield chunk.choices[index].delta.content

    def get_choice_content(self, choice):
        return choice.message.content

    def get_finish_reason(self, choice):
        reason = choice.finish_reason
        finish_reasons = {
            'stop': 'STOP',
            'length': 'MAX_TOKENS',
            'content_filter': 'SAFETY',
            'function_call': 'FUNCTION_CALL',
            'tools_call': 'TOOL_CALL',
            'null': 'NULL'
        }
        return finish_reasons.get(reason, 'UNKNOWN')

    def is_choice_safe(self, index=0) -> bool:
        choice = self.get_choice(index)
        if self.streamed:
            return True
        return choice.finish_reason == 'stop'


class GeminiResponse(Response):
    def __init__(self, response: 'genai.protos.GenerateContentResponse', streamed: bool, generator: 'ResponseGenerator', prompt: 'BasePrompt'):
        super().__init__(response, streamed, generator, prompt)
        
    def _get_choices(self) -> List:
        return self._response.candidates if hasattr(self._response, 'candidates') else [self._response]

    def stream_text(self, index=0) -> Generator[str, None, None]:
        for chunk in self._response:
            yield chunk
    
    def get_choice_content(self, choice):
        return choice.content.parts[0].text

    def get_finish_reason(self, choice: genai.types.protos.Candidate):
        reason = choice.finish_reason.value
        for attr_name, attr_value in genai.types.protos.Candidate.FinishReason.__dict__.items():
            if attr_value == reason:
                print(attr_value)
                return attr_name

    def is_choice_safe(self, index=0) -> bool:
        choice = self.get_choice(index)
        return choice.finish_reason == genai.types.protos.Candidate.FinishReason.STOP