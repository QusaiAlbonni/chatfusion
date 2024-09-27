from abc import ABC, abstractmethod
from typing import Iterable, List
from .prompts.prompts import BasePrompt, SingleMessagePrompt, ChatPrompt, Text, File, Part, SystemMessage
from .model_registry import genai, openai
from .responses import OpenAIResponse, GeminiResponse, Response
from .exceptions import MissingLMLibs, BadInputException


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


class PromptStrategy(ABC):
    @abstractmethod
    def serialize(self, prompt: 'BasePrompt') -> any:
        pass


class GeminiGenerator(ResponseGenerator, PromptStrategy):
    def __init__(self, model_name, temperature=0.7, **kwargs):
        self.response = None
        if genai is None:
            raise MissingLMLibs(
                "Missing Gemini Libs, install google's generativeai")
        self.model = genai.GenerativeModel(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.response = None

    def generate_response(self, prompt: 'BasePrompt', *args, **kwargs) -> GeminiResponse:
        from google.api_core.retry import Retry
        from google.generativeai.generative_models import helper_types

        candidate_count = kwargs.pop('choice_count', 1)
        temperature = kwargs.pop('temperature', self.temperature)
        retry = kwargs.pop('retry', False)

        contents, system_instructions = prompt.build_prompt(self)
        if system_instructions:
            self.include_system_instructions(system_instructions)

        if retry:
            retry = Retry()
        else:
            retry = None

        response = self.model.generate_content(
            contents=contents,
            generation_config=genai.GenerationConfig(
                temperature=temperature, candidate_count=candidate_count, *args, **kwargs),
            request_options=helper_types.RequestOptions(retry=retry),
            *args,
            **kwargs
        )

        self.streamed = kwargs.pop('stream', False)
        self.response = response
        return GeminiResponse(response, self.streamed, self, prompt)

    def set_temperature(self, temperature):
        self.temperature = temperature
        genai.GenerationConfig.temperature = temperature

    def serialize(self, prompt: 'BasePrompt') -> tuple:
        contents = prompt.get_content()
        final = []
        system_instructions = []
        if isinstance(prompt, SingleMessagePrompt):
            final = self.serialize_many_parts(contents)
            return final, system_instructions
        if isinstance(prompt, ChatPrompt):
            contents = prompt.get_content()
            for message in contents:
                content = self.serialize_many_parts(message.content)
                messagedict = {
                    'role': self.get_appropriate_role(message.role),
                    'parts': content
                }
                if message.role != "system":
                    final.append(messagedict)
                    continue
                system_instructions.append(message)
            return final, system_instructions

    def get_appropriate_role(self, role: str) -> str:
        if role == 'assistant':
            return 'model'
        else:
            return 'user'

    def serialize_one_part(self, part: Part):
        if isinstance(part, Text):
            return part.content
        elif isinstance(part, File):
            return self.handle_file(part)
        else:
            raise ValueError("Something went wrong with type checking")

    def handle_file(self, file: File):
        from google.api_core.exceptions import PermissionDenied

        if file.inline:
            return {'mime_type': file.type, 'data': file.data}
        else:
            try:
                gemini_file = genai.get_file(file.id)
            except PermissionDenied:
                gemini_file = genai.upload_file(
                    file.get_path(), mime_type=file.type, name=file.id)
            return gemini_file

    def serialize_many_parts(self, parts: Iterable[Part]):
        l = []
        if not isinstance(parts, Iterable):
            return self.serialize_one_part(parts)
        for part in parts:
            item = self.serialize_one_part(part)
            l.append(item)
        return l

    def include_system_instructions(self, system_instructions: List[SystemMessage]):
        temp = []
        for instruction in system_instructions:
            instruction_content = self.serialize_many_parts(
                instruction.get_content())
            temp += instruction_content if isinstance(instruction_content, list) else [
                instruction_content]
        self.model._system_instruction = genai.types.content_types.to_content(
            instruction_content)


class OpenAiGenerator(ResponseGenerator, PromptStrategy):
    def __init__(self, model_name, temperature=0.7, **kwargs):
        if not openai:
            raise MissingLMLibs("Missing OpenAI Libs, install openai package")
        self.model_name = model_name
        self.temperature = temperature
        self.response = None
        self.client = openai.OpenAI(**kwargs)
        self.streamed = False

    def generate_response(self, prompt: 'BasePrompt', *args, **kwargs) -> OpenAIResponse:
        candidate_count = kwargs.pop('choice_count', 1)
        temperature = kwargs.pop('temperature', self.temperature)
        retry = kwargs.pop('retry', False)

        contents = prompt.build_prompt(self)

        if isinstance(prompt, SingleMessagePrompt):
            method = self.client.completions
            kwargs['prompt'] = contents
        else:
            method = self.client.chat.completions
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

    def set_temperature(self, temperature):
        self.temperature = temperature

    def serialize(self, prompt: 'BasePrompt') -> list:
        contents = prompt.get_content()
        final = []

        if isinstance(prompt, SingleMessagePrompt):
            final = self.serialize_many_parts(contents)
        elif isinstance(prompt, ChatPrompt):
            for message in contents:
                content = self.serialize_many_parts(message.content)
                final.append({
                    "role": self.get_appropriate_role(message.role),
                    "content": content
                })

        return final

    def get_appropriate_role(self, role: str) -> str:
        if role == "system":
            return "system"
        elif role == "assistant":
            return "assistant"
        else:
            return "user"

    def serialize_one_part(self, part: Part):
        if isinstance(part, Text):
            return part.content
        elif isinstance(part, File):
            return self.handle_file(part)
        else:
            raise ValueError("Something went wrong with type checking")

    def handle_file(self, file: File):
        if not file.type.startswith("image"):
            raise BadInputException(
                "File is not an image", "Only images are supported for file uploads in openai")
        if file.inline:
            return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{file.base64_data}"}}
        return {'type': 'image_url', 'image_url': {'url': file.uri}}

    def serialize_many_parts(self, parts: Iterable[Part]) -> str:
        l = []
        if not isinstance(parts, Iterable):
            return self.serialize_one_part(parts)
        for part in parts:
            item = self.serialize_one_part(part)
            l.append(item)
        return l
