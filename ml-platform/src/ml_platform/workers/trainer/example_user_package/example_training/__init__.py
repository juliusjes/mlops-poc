
from .loop import MyLoop
from .model import SimpleTestModel, SimpleResNet

PLUGIN = {
    "model": {
        "simple_test_model": SimpleTestModel,
        "simple_resnet": SimpleResNet
        },
    "loop": {
        "myloop": MyLoop
        },
    "data_source": None,
}