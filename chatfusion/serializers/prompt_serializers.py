from chatfusion.prompts.prompts import BasePrompt
from .base import Serializer, AsyncSerializer
from ..prompts.prompts import BasePrompt, SingleMessagePrompt, ChatPrompt, Text, File, Part, SystemMessage, Message
from typing import Iterable, List
from ..model_registry import genai, openai
from ..exceptions import InvalidPrompt
from functools import lru_cache

class OpenAiSerializer(Serializer):
    @lru_cache
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
            return {'type': 'text', 'text': part.content}
        elif isinstance(part, File):
            return self.handle_file(part)
        else:
            raise ValueError("Something went wrong with type checking")

    def handle_file(self, file: File):
        if not file.type.startswith("image"):
            raise InvalidPrompt(
                "File is not an image", "Only images are supported for file uploads in openai")
        if file.inline:
            return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{file.base64_data}"}}
        return {'type': 'image_url', 'image_url': {'url': file.uri}}

    def serialize_many_parts(self, parts: Iterable[Part]) -> str:
        l = []
        if not isinstance(parts, Iterable):
            l.append(self.serialize_one_part(parts))
            return l
        for part in parts:
            item = self.serialize_one_part(part)
            l.append(item)
        return l
    
    def get_system_instructions(self, prompt: BasePrompt) -> any:
        return []

class GeminiSerializer(Serializer):
    @lru_cache
    def serialize(self, prompt: 'BasePrompt') -> list:
        contents = prompt.get_content()
        final = []
        if isinstance(prompt, SingleMessagePrompt):
            final = self.serialize_many_parts(contents)
            return final
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
            return final

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
    
    @lru_cache
    def get_system_instructions(self, prompt) -> any:
        if isinstance(prompt, SingleMessagePrompt):
            return None
        temp = []
        contents: list[Message] = prompt.get_content()
        for message in contents:
            if message.role != 'system':
                continue
            instruction_content = self.serialize_many_parts(
                message.get_content())
            temp += instruction_content if isinstance(instruction_content, list) else [
                instruction_content]
        return temp