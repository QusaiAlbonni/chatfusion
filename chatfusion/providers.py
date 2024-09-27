from .types import Dict, Any, Optional, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from .generators import ResponseGenerator
    
class Provider:
    def __init__(self, name: str, default_model: str = '', initial_models: Dict[str, Any] = None, api_key: str = ''):
        self.name = name
        self.models: Dict[str, Any] = initial_models or {}
        self.default_model = default_model
        self.generator: Optional[Type['ResponseGenerator']] = None

    def set_model(self, model_name: str, model_data: Any):
        self.models[model_name] = model_data

    def get_model(self, model_name: str) -> Optional[Any]:
        return self.models.get(model_name)

    def delete_model(self, model_name: str):
        self.models.pop(model_name, None)

    def set_default_model(self, model_name: str):
        self.default_model = model_name

    def set_generator(self, generator: Type['ResponseGenerator']):
        self.generator = generator

    def get_generator(self) -> Optional[Type['ResponseGenerator']]:
        return self.generator
