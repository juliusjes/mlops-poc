import json
import logging
import time
from typing import Optional

import pandas as pd
import requests
import torch
from ml_platform.inference_server.collector import Collector
from ml_platform.inference_server.config import CollectorConfig, PluginConfig
from ml_platform.inference_server.observation import Observation
from ml_platform.inference_server.plugins.drift_detection.detectors.c2st import (
    C2STDriftMethod,
)
from ml_platform.inference_server.plugins.drift_detection.detectors.conformal import (
    Conformal2,
)
from ml_platform.inference_server.plugins.drift_detection.detectors.energydistance import (
    EnergyDistanceDriftMethod,
)
from ml_platform.inference_server.plugins.drift_detection.detectors.mmd import (
    MMDDriftMethod,
)
from ml_platform.inference_server.plugins.plugin import Plugin, PluginResult
from ml_platform.inference_server.plugins.plugin_logger import PluginLogger
from ml_platform.shared.tracing import _TRACE, enable_tracing
from ml_platform.shared.utils import np_encoder

logger = logging.getLogger(__name__)


drift_detector_registry = {
    "conformal": Conformal2,
    "mmd": MMDDriftMethod,
    "energydistance": EnergyDistanceDriftMethod,
    "c2st": C2STDriftMethod,
}


class DriftDetector(Plugin):
    def __init__(self, config: PluginConfig, collector: Collector):
        super().__init__(config, collector)

        self.window_size = 100
        self.current_window: list[Observation] = []
        self.reference_window: list[Observation] = []
        self.frequency = 10
        self.counter = 0

        cls = drift_detector_registry.get(self.config.name, None)
        if cls is None:
            raise KeyError(f"Invalid drift detector name in config ({self.config.name})")

        # init
        if len(config.trace_fields) > 0:
            self.detector = enable_tracing(cls(**self.config.parameters), self.config.trace_fields)
        else:
            self.detector = cls(**self.config.parameters)


        self.notifiers = self.config.notifiers
        self.reset_on_detection = self.config.reset_on_detection
        self.reference_window_update = self.config.reference_window_update


    def update_window(self):
        if self.window_size % self.reference_window_update == 0:
            self.reference_window = []

        if len(self.reference_window) < self.window_size:
            self.reference_window = self.collector.get_n_latest_observations(n=self.window_size)

        self.current_window = self.collector.get_n_latest_observations(n=self.window_size)


    def _notify(self, data: dict):
        """
        Send requests to every endpoint in the list.
        """
        if not isinstance(data, dict):
            data = {}

        data = json.dumps(data, default=np_encoder)

        for endpoint in self.notifiers:
            try:
                r = requests.post(url=endpoint, json=data)
                if r.status_code != 200:
                    logger.warning( "%s returned non-200 status code %s", endpoint, r.status_code, )

            except requests.RequestException as e:
                logger.warning("Failed to notify %s: %s", endpoint, e)


    def get_embeddings(self, observations: list[Observation]):
        embs = []
        for o in observations:
            if isinstance(o.prediction, tuple):
                emb = o.prediction[1].reshape((-1))
                if isinstance(emb, torch.Tensor):
                    emb = emb.numpy()
                embs.append(emb)
            elif isinstance(o.prediction, dict):
                emb = o.prediction["emb"].reshape((-1))
                if isinstance(emb, torch.Tensor):
                    emb = emb.numpy()
                embs.append(emb)
            else:
                raise ValueError(f"Invalid prediction format")

        return embs

    def process(self) -> Optional[PluginResult]:
        self.counter += 1
        if self.counter % self.frequency != 0:
            return

        self.update_window()
        if not len(self.reference_window) >= self.window_size:
            return

        current_embeddings = pd.DataFrame(self.get_embeddings(self.current_window))
        reference_embeddings = pd.DataFrame(self.get_embeddings(self.reference_window))

        value, is_drift, name = self.detector(current_embeddings, reference_embeddings)

        if is_drift:
            self._notify(data={"name": name, "value": value, "is_drift": is_drift})

            if self.reset_on_detection:
                self.reference_window = []
                self.current_window = []
                if hasattr(self.detector, "_reset"):
                    self.detector._reset()


        results = {
            "time_stamp": self.current_window[-1].time_stamp,
            "sample_index": self.current_window[-1].sample_index,
        }

        for tf in self.config.trace_fields:
            value = _TRACE.get(self.detector.get_fingerprint())["locals"][tf]
            results[tf] = value

        #self.logger.collect(results, self.config.name)
        return PluginResult(name=self.config.name, metrics=results)



async def test(dd):
    return await dd.process()



if __name__ == "__main__":
    
    import asyncio

    import numpy as np

    ccfg = CollectorConfig()
    pcfg = PluginConfig(
        plugin_type="drift",
        name="energydistance",
        parameters={
            "bootstrap": True
        },
        trace_fields=["res", "perc"],
    )

    pl = PluginLogger()
    pl.init_log()

    def ct():
        return time.time()

    def td(t0, t1=None):
        if t1 is None: t1=time.time()
        return t1-t0

    collector = Collector(ccfg)

    N = 128
    dd = DriftDetector(pcfg, collector)

    tds = np.zeros((1000))


    async def test():
        times = []

        for i in range(1000):
            obs = Observation(
                input=np.random.uniform(size=(150,150)),
                prediction={"logits": [0.5, 0.5], "emb": np.random.uniform(size=(N))},
                label=1.0,
                sample_index=i,
            )
            
            collector.collect(obs)
            t0 = time.perf_counter()

            r = await asyncio.to_thread(dd.process)

            if r is not None:
                pl.collect(r)
                print("c")

            times.append(time.perf_counter() - t0)

        print(np.mean(times))
        print(np.median(times))
        print(np.max(times))

    asyncio.run(test())


    # import matplotlib.pyplot as plt

    # plt.hist(tds[(tds < 1) & (tds > 0.001)], bins=100)
    # plt.show()



