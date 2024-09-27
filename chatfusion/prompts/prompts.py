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



class Prompt(BasePrompt):
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

    def message(self, role, content: Content):
        return ChatPrompt(self.parts + [Message(role ,content)])
    
    def messages(self, messages: Iterable[DictMessage]):
        l = [Message(message['role'], message['content']) for message in messages ]
        return ChatPrompt(self.parts + l)

    def user(self, content: Content):
        return ChatPrompt(self.parts + [UserMessage(content)])

    def assistant(self, content: Content):
        return ChatPrompt(self.parts + [AssistantMessage(content)])
    
    def system(self, content: Content):
        message = SystemMessage(content)
        return ChatPrompt(self.parts + [message])
    
    def get_content(self) -> list[Message]:
        return super().get_content()
