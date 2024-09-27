from typing import Iterable, Union, Literal, TypedDict, Protocol, runtime_checkable, Optional, Any, BinaryIO, TextIO, IO, TYPE_CHECKING, List, Tuple, Type, Dict, Generator, Callable, ParamSpec
from io import IOBase

RoleType = Literal['user', 'system']

File = Union[IO[str], TextIO, BinaryIO]

FileTypeCheck = IOBase

Content = Union[Iterable[Union[str, File]], File, str]

class Message(TypedDict):
    role: RoleType
    content: Content


MessageList = Iterable[Message]


@runtime_checkable
class LMSerializable(Protocol):
    """Protocol for objects that can be serialized into messages for language models.

    Classes implementing this protocol can be easily converted into a format
    suitable for communication with various language model providers.

    The to_message method should return a representation of the object
    that can be used as input for a language model, tailored to the specified provider.
    """

    def to_message(self, provider: str, role: Optional[Literal['user', 'system']] = None) -> Any:
        """Convert the object to a message format for the specified LM provider.

        Args:
            provider (str): The name of the language model provider (e.g., 'openai', 'gemini', 'anthropic').

        Returns:
            Any: A representation of the object suitable for the specified provider.
        """
        ...

