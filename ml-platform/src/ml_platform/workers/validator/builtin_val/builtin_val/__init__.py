from .context import BasicContext
from .data_sources.lakefs_source import LakeFSSource
from .loops.validation_loop import PyTorchClassifierValidationLoop

PLUGIN = {
    "model": None,
    "data_source": {
        "lakefs_source": LakeFSSource
    },
    "context": {
        "basic_context": BasicContext
    },
    "loop": {
        "pytorch_classification_validation_loop": PyTorchClassifierValidationLoop
    }
}