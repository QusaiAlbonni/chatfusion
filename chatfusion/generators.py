from abc import ABC, abstractmethod
from typing import Any, Coroutine, Iterable, List, TYPE_CHECKING, Awaitable, TypedDict

from chatfusion.responses import Response
from .prompts.prompts import BasePrompt, SingleMessagePrompt, ChatPrompt, Text, File, Part, SystemMessage
from .model_registry import genai, openai
from .responses import OpenAIResponse, GeminiResponse, Response
from .exceptions import InvalidPrompt, MissingBackends
from .serializers import Serializer, OpenAiSerializer, GeminiSerializer


class ResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, prompt, **kwargs) -> Response:
        """
        this method is the main method of generating your content based on a passed Prompt object
        this method may fail due to provider specific or model specific reasons 

        Args:
            prompt (BasePrompt):
                a Prompt object that will be used to generate the response not all prompts will work for all models


        Returns:
            Response: 
                a response object in which you can access the data returned 

        **kwargs:
            temperature (float): will override the default
            choice_count (int): how many times should the model generate the content (model/subscription specific) may fail if more than 1 
            retry (bool): whether or not to retry on failure
        """
        pass

class AsyncResponseGenerator(ABC):
    @abstractmethod
    async def agenerate_response(self, prompt, *args, **kwargs) -> Response:
        """
        Async version of response generation
        
        Warning: Does not yet support streamed responses
        
        this method is the main method of generating your content based on a passed Prompt object
        this method may fail due to provider specific or model specific reasons 

        Args:
            prompt (BasePrompt):
                a Prompt object that will be used to generate the response not all prompts will work for all models


        Returns:
            Response: 
                a response object in which you can access the data returned 

        **kwargs:
            temperature (float): will override the default
            choice_count (int): how many times should the model generate the content (model/subscription specific) may fail if more than 1 
            retry (bool): whether or not to retry on failure
        """
        pass
    



class GeminiGenerator(ResponseGenerator, AsyncResponseGenerator):
    model: genai.GenerativeModel
    model_name: str
    temperature: float
    response: GeminiResponse
    serializer: Serializer
    
    class GenerationArgs(TypedDict):
        generation_config: genai.GenerationConfig
        contents: genai.types.content_types.ContentsType
        request_options: any
    
    def __init__(self, model_name, temperature=0.7, prompt_serializer= GeminiSerializer(), **kwargs):
        self.response = None
        if genai is None:
            raise MissingBackends(
                "Missing Gemini Libs, install google's generativeai")
        self.model = genai.GenerativeModel(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.response = None
        self.serializer= prompt_serializer

    def generate_response(self, prompt: 'BasePrompt', *args, **kwargs) -> GeminiResponse:
        generation_args = self._prepare_generation_args(prompt, *args, **kwargs)
        
        response = self.model.generate_content(
            contents=generation_args['contents'],
            generation_config=generation_args['generation_config'],
            request_options=generation_args['request_options'],
        )

        self.streamed = kwargs.pop('stream', False)
        self.response = response
        return GeminiResponse(response, self.streamed, self, prompt)
    
    async def agenerate_response(self, prompt, *args, **kwargs) -> Awaitable[Response]:
        generation_args = self._prepare_generation_args(prompt, *args, **kwargs)
        
        response = await self.model.generate_content_async(
            contents=generation_args['contents'],
            generation_config=generation_args['generation_config'],
            request_options=generation_args['request_options'],
        )

        self.streamed = kwargs.pop('stream', False)
        self.response = response
        return GeminiResponse(response, self.streamed, self, prompt)

    def set_temperature(self, temperature):
        self.temperature = temperature
        genai.GenerationConfig.temperature = temperature
    
    def _call_api(self, contents, generation_config, request_options, *args, **kwargs):
        return self.model.generate_content(
            contents= contents,
            generation_config= generation_config,   
        )
        
    def _prepare_generation_args(self, prompt: 'BasePrompt', *args, **kwargs) -> GenerationArgs:
        from google.api_core.retry import Retry
        from google.generativeai.generative_models import helper_types

        candidate_count = kwargs.pop('choice_count', 1)
        temperature = kwargs.pop('temperature', self.temperature)
        retry = kwargs.pop('retry', False)

        contents = prompt.build_prompt(self.serializer)
        system_instructions = self.serializer.get_system_instructions(prompt)
        if system_instructions:
            self._include_system_instructions(system_instructions)

        if retry:
            retry = Retry()
        else:
            retry = None
            
        return {
            'generation_config' : genai.GenerationConfig(
                candidate_count=candidate_count,
                temperature=temperature,     
            ),
            'request_options': helper_types.RequestOptions(retry=retry),
            'contents': contents,
        }
    def _include_system_instructions(self, system_instructions: List[SystemMessage]):
        self.model._system_instruction = genai.types.content_types.to_content(
            system_instructions)
        


class OpenAiGenerator(ResponseGenerator, AsyncResponseGenerator):
    model_name: str
    temperature: float
    response: OpenAIResponse
    client: openai.OpenAI
    serializer: Serializer
    streamed: bool
    
    def __init__(self, model_name, temperature=1.0, prompt_serializer: Serializer = OpenAiSerializer(), **kwargs):
        if not openai:
            raise MissingBackends("Missing OpenAI Libs, install openai package")
        self.model_name = model_name
        self.temperature = temperature
        self.response = None
        self.client = openai.OpenAI(**kwargs)
        self.client_async = openai.AsyncOpenAI(**kwargs)
        self.streamed = False
        self.serializer= prompt_serializer
        
    def generate_response(self, prompt: 'BasePrompt', *args, **kwargs) -> OpenAIResponse:
        candidate_count = kwargs.pop('choice_count', 1)
        temperature = kwargs.pop('temperature', self.temperature)
        retry = kwargs.pop('retry', False)
        chat_completions_only= kwargs.pop('chat_completions_only', True)

        contents = prompt.build_prompt(self.serializer)

        if isinstance(prompt, SingleMessagePrompt) and not chat_completions_only:
            method = self.client.completions
            kwargs['prompt'] = contents
        else:
            method = self.client.chat.completions
            if isinstance(prompt, SingleMessagePrompt):
                kwargs['messages'] = [{
                    'role': 'user',
                    'content': contents
                }]    
            else:
                kwargs['messages'] = contents

        response = method.create(
            model=self.model_name,
            temperature=temperature,
            n=candidate_count,
            *args,
            **kwargs
        )

        self.streamed = kwargs.get('stream', False)
        self.response = response
        return OpenAIResponse(response, self.streamed, self, prompt)

    async def agenerate_response(self, prompt: 'BasePrompt', *args, **kwargs) -> OpenAIResponse:
        candidate_count = kwargs.pop('choice_count', 1)
        temperature = kwargs.pop('temperature', self.temperature)
        retry = kwargs.pop('retry', False)
        chat_completions_only= kwargs.pop('chat_completions_only', True)

        contents = prompt.build_prompt(self.serializer)

        if isinstance(prompt, SingleMessagePrompt) and not chat_completions_only:
            method = self.client_async.completions
            kwargs['prompt'] = contents
        else:
            method = self.client_async.chat.completions
            if isinstance(prompt, SingleMessagePrompt):
                kwargs['messages'] = [{
                    'role': 'user',
                    'content': contents
                }]    
            else:
                kwargs['messages'] = contents

        response = await method.create(
            model=self.model_name,
            temperature=temperature,
            n=candidate_count,
            *args,
            **kwargs
        )

        self.streamed = kwargs.get('stream', False)
        self.response = response
        return OpenAIResponse(response, self.streamed, self, prompt)

    def set_temperature(self, temperature):
        self.temperature = temperature

    
