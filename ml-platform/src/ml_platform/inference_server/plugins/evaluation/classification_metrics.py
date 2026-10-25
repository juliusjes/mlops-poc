import logging
from typing import Optional

import numpy as np
import torch
from ml_platform.inference_server.observation import Observation
from ml_platform.inference_server.plugins.plugin import Plugin, PluginResult
from sklearn.metrics import accuracy_score, f1_score, recall_score

logger = logging.getLogger(__name__)

class ClassificationMetrics(Plugin):
    """
    Classification metrics that require labels
    """
    def __init__(self, config, collector):
        super().__init__(config, collector)

        self.window_size = 50
        self.window: list[Observation] = []
        self.frequency = 20
        self.counter = 0

    def update_window(self):
        self.window = self.collector.get_n_latest_observations(n=self.window_size)


    def predictions_to_numpy(self, predictions):
        arrays = []

        for yp in predictions:
            if isinstance(yp, tuple):
                yp = yp[0]
            elif isinstance(yp, dict):
                yp = yp["logits"]

            if isinstance(yp, torch.Tensor):
                yp = yp.detach().cpu().numpy()

            yp = np.asarray(yp)

            # Make single predictions (2,) into (1, 2)
            if yp.ndim == 1:
                yp = yp[None, :]

            arrays.append(yp)

        logits = np.concatenate(arrays, axis=0)
        return self.logits_to_label(logits)

    def logits_to_label(self, logits: np.ndarray):
        """
        Convert logits of shape [samples, classes] to predicted class labels.
        """
        return np.argmax(logits, axis=1)


    def process(self)-> Optional[PluginResult]: # TODO: consider adding Expected calibaration error
        self.counter += 1
        if self.counter % self.frequency != 0:
            return

        self.update_window()

        label_mask = np.array([o.label != None for o in self.window])
        y_true = np.array([o.label for o in self.window])[label_mask]
        y_pred = [o.prediction for o in self.window]
        y_pred = self.predictions_to_numpy(y_pred)[label_mask]

        time_stamp = self.window[-1].time_stamp
        sample_index = self.window[-1].sample_index

        metrics = {
            "time_stamp": time_stamp,
            "sample_index": sample_index,
            f"accuracy_rw{self.window_size}": float(accuracy_score(y_true, y_pred)),
            f"recall_rw{self.window_size}": float(recall_score(y_true, y_pred)),
            f"f1_rw{self.window_size}": float(f1_score(y_true, y_pred))
        }

        #self.logger.collect(metrics, self.config.name)
        return PluginResult(name=self.config.name, metrics=metrics)
        
# if __name__ == "__main__":

#     from ml_platform.inference_server.collector import Collector, CollectorConfig
#     from ml_platform.inference_server.config import PluginConfig
#     from ml_platform.inference_server.observation import Observation
#     pcfg = PluginConfig(
#         plugin_type="eval",
#         name="test",
#         parameters={},
#         trace_fields=[],
#     )

#     ccfg = CollectorConfig()
#     c = Collector(ccfg)
#     m = ClassificationMetrics(config=pcfg, collector=c)


#     for i in range(100):
#         print(i)
#         o = Observation(
#             input=torch.rand(10),                    # Uniform in [0, 1)
#             prediction=torch.rand(2),                # Two logits/scores
#             label=np.random.randint(0, 2),         # Random label: 0 or 1
#             sample_index=i,
#         )

#         c.collect(o)
#         r = m.process()
#         print(r)

    