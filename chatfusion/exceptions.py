class BadInputException(Exception):
    def __init__(self, message: str, reason: str):
        self.message = message
        self.reason = reason
        message = f"Invalid LM input prompt: {self.message}. Reason: {self.reason}\n check the specific model safety guidlines and other specification"
        super().__init__(self.message + " " + self.reason)

class ModelNotFoundException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
class MissingLMLibs(ImportError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
class UnexpectedBehavior(Exception):
    def __init__(self, message: str, behavior: str) -> None:
        super().__init__(message + f" the behavior {behavior} was unexpected")
        
class ForbiddenException(Exception):
    def __init__(self, *args: object) -> None:
        newargs = ['Forbidden action performed'] + list(args)
        super().__init__(*newargs)