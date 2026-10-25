from .loops.pytorch_classification_loop import PyTorchClassificationLoop
from .data_sources.lakefs_source import LakeFSSource
from .context.basic_context import BasicContext
PLUGIN = {
    "model": None,
    "loop": {
        "pytorch_classification_loop": PyTorchClassificationLoop,
    },
    "data_source": {
        "lakefs_source": LakeFSSource
    },
    "context": {
        "basic_context": BasicContext
    }
}