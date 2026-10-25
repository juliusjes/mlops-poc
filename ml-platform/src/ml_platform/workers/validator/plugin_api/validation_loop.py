from pydantic import BaseModel
from .context import ValidationContext


class ValidationResult:
    def __init__(self, validation_info= None):
        self.validation_info: dict|None = validation_info

    def to_dict(self):
        return {
            k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v
            for k, v in self.validation_info.items()
        }
            

class ValidationLoop:
    def __init__(self):
        pass

    def validate(self, ctx: ValidationContext) -> ValidationResult:

        return ValidationResult()
            


