from dataclasses import dataclass

from ml_platform.inference_server.collector import Collector
from ml_platform.inference_server.config import PluginConfig


@dataclass
class PluginResult:
    name: str
    metrics: dict

class Plugin():
    def __init__(self, config: PluginConfig, collector: Collector):
        self.config = config
        self.collector = collector

    def process(self):
        # run calculations using data from collector
        return PluginResult(name="plugin", metrics={})