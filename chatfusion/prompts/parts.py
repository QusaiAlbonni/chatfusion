from __future__ import annotations

from mimetypes import guess_type
from typing import TYPE_CHECKING
from ..types import Content, FileTypeCheck, Iterable, Message as DictMessage, File as FileType
from uuid import uuid4
import base64


class Part:
    def __init__(self, content: any) -> None:
        self.content = content

    def __str__(self) -> str:
        return str(self.content)
    
    def get_data(self):
        return self.content
    
class PartConvertableMixin:
    def to_part(self, content: Content) -> Part:
        if isinstance(content, str):
            return Text(content)
        elif isinstance(content, FileTypeCheck):
            return File(content)
        elif isinstance(content, File):
            return content
        elif isinstance(content, Text):
            return content
        elif isinstance(content, Iterable):
            temp = []
            for item in content:
                if isinstance(item, str):
                    temp.append(Text(item))
                elif isinstance(item, Iterable):
                    temp.append(self.to_part(item))
                elif isinstance(item, File):
                    temp.append(item)
                elif isinstance(item, Text):
                    temp.append(item)
                else:
                    raise ValueError(f"Invalid content type: {type(item)}")
            return temp
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

class PartStringifyMixin:
    def to_str(self, parts: Part):
        if isinstance(parts, Iterable):
            temp = []
            for part in parts:
                string = str(part)
                temp.append(string)
            return str(temp)
        return str(parts)

class Message(Part, PartConvertableMixin, PartStringifyMixin):
    def __init__(self, role, content: Content) -> None:
        content= self.to_part(content)
        super().__init__(content)
        self.role = role

    def get_role(self) -> str:
        return self.role

    def get_content(self) -> Content:
        return self.content

    def to_dict(self) -> DictMessage:
        return {'role': self.role, 'content': self.content}

    def __str__(self) -> str:
        return f"{self.get_role()}: {self.to_str(self.content)}"


class UserMessage(Message):
    def __init__(self, content: Content) -> None:
        super().__init__('user', content)

    @classmethod
    def from_text(cls, text: str) -> UserMessage:
        return cls(Text(text))

    @classmethod
    def from_file(cls, file: FileType) -> UserMessage:
        return cls(File(file))


class SystemMessage(Message):
    def __init__(self, content: Content) -> None:
        super().__init__('system', content)

    @classmethod
    def from_text(cls, text: str) -> SystemMessage:
        return cls(Text(text))



class AssistantMessage(Message):
    def __init__(self, content: Content) -> None:
        super().__init__('assistant', content)

    @classmethod
    def from_text(cls, text: str) -> AssistantMessage:
        return cls(Text(text))


class Text(Part):
    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string got {type(text)}")
        super().__init__(text)

    def __str__(self) -> str:
        return self.content
    
    def get_data(self):
        return self.text


class File(Part):
    def __init__(self, file: FileType = None, inline: bool= False, file_type: str=None, uri: str= None, id: str= None) -> None:
        if not isinstance(file, FileTypeCheck):
            raise ValueError(f"File must be an IOBase or a subclass of it got {type(file)}")
        self.uri = uri
        self.id = str(id or uuid4())
        self._file = file
        self.inline = inline
        if file is not None and inline:
            file.seek(0)
            self.data = file.read()
            self.base64_data = base64.b64encode(self.data).decode('utf-8')
        if file is None and uri is None:
            raise ValueError("File data or uri must be provided")
        if file_type is None:
            self.type, _ = guess_type(file.name)
        else:
            self.type = file_type
        super().__init__(self._file)

    def __str__(self) -> str:
        return f"File: {self.content.name}, Type: {self.type}"
    
    def get_data(self):
        return self.data
    
    def get_path(self):
        return self._file.name
    
    def get_file_object(self):
        return self._file

