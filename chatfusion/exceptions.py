class InvalidPrompt(Exception):
    def __init__(self, message: str, reason: str):
        self.message = message
        self.reason = reason
        message = f"Invalid input prompt: {self.message}. Reason: {self.reason}\n check the specific model safety guidlines and other specification"
        super().__init__(self.message + " " + self.reason)

class ModelNotFound(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
class MissingBackends(ImportError):
    def __init__(self, *args: object) -> None:
        super().__init__("Missing Generation backend for AI, check if you have installed packages responsible for generation.",*args)
        
class UnexpectedBehavior(Exception):
    def __init__(self, message: str, behavior: str) -> None:
        super().__init__(message + f" the behavior {behavior} was unexpected")
        
class Forbidden(Exception):
    def __init__(self, *args: object) -> None:
        newargs = ['Forbidden action performed'] + list(args)
        super().__init__(*newargs)