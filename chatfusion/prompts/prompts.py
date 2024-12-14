from __future__ import annotations

from ..types import Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    from ..types import Message as DictMessage, Content
from ..types import File as FileType
from .parts import Part, Text, File, Message, SystemMessage, UserMessage, AssistantMessage


class BasePrompt:

    def __init__(self, parts: list[Part] | Part | None = None) -> None:
        self.parts = parts if isinstance(parts, list) else [
            parts] if parts else []

    def __str__(self) -> str:
        return "\n'''\n" + "\n".join(str(part) for part in self.parts) + "\n'''"

    def build_prompt(self, prompt_strategy) -> any:
        implementaion_prompt = prompt_strategy.serialize(self)
        return implementaion_prompt

    def get_content(self):
        return self.parts

class TranslationMixin:
    def translate(
        self,
        text: str,
        context: str= "",
        lang_from: str= None,
        lang_to: str=None,
        style: str=None,
        format: str=None,
        extra: str=None
        ):
        
        prompt = "Translate the following text "
        if lang_from:
            prompt += f"from {lang_from} "
        if lang_to:
            prompt += f"to {lang_to} "
        if style:
            prompt += f"{style}"
        if format:
            prompt += f"format the output as {format}"
        
        if context:
            prompt += f"with the context:\n {context} "
        
        prompt += f"text is: '{text}' \n"
        
        if extra:
            prompt += extra
            
        
        return SingleMessagePrompt().text(prompt)
        

class Prompt(BasePrompt, TranslationMixin):
    def __init__(self, parts: list[Part] | Part | None = None) -> None:
        super().__init__(parts if isinstance(
            parts, list) else [parts] if parts else [])

    def text(self, text: str | Text):
        if isinstance(text, Text):
            part = text
        else:
            part = Text(text)
        return SingleMessagePrompt(self.parts + [part])

    def file(self, file: FileType | File):
        if isinstance(file, File):
            part = file
        else:
            part = File(file)
        return SingleMessagePrompt(self.parts + [part])

    def chat(self):
        return ChatPrompt(self.parts)


class SingleMessagePrompt(BasePrompt):
    def __init__(self, parts: list[Part] | Part = None) -> None:
        super().__init__(parts)

    def text(self, text: str | Text):
        if isinstance(text, Text):
            part = text
        else:
            part = Text(text)
        return SingleMessagePrompt(self.parts + [part])

    def file(self, file: FileType | File):
        if isinstance(file, File):
            part = file
        else:
            part = File(file)
        return SingleMessagePrompt(self.parts + [part])


class ChatPrompt(BasePrompt):
    
    parts: list[Message]
    
    def __init__(self, parts: list[Part] | Part | None = None) -> None:
        super().__init__(parts if isinstance(
            parts, list) else [parts] if parts else [])
        
    def delete(self, index):
        parts = self.parts
        return ChatPrompt([part for i, part in enumerate(parts) if i != index])

    def delete_by_id(self, id):
        parts = self.parts
        return ChatPrompt(list(filter(lambda element: element.id != id, parts)))
    
    def message(self, role, content: Content, id=None):
        return ChatPrompt(self.parts + [Message(role ,content, id)])
    
    def messages(self, messages: Iterable[DictMessage]):
        l = [Message(message['role'], message['content'], message.get('id', None)) for message in messages ]
        return ChatPrompt(self.parts + l)

    def user(self, content: Content, id=None):
        return ChatPrompt(self.parts + [UserMessage(content, id)])

    def assistant(self, content: Content, id=None):
        return ChatPrompt(self.parts + [AssistantMessage(content, id)])
    
    def system(self, content: Content, id=None):
        message = SystemMessage(content, id)
        return ChatPrompt(self.parts + [message])
    
    def get_content(self) -> list[Message]:
        return super().get_content()

