from evidently.metrics.embeddings import MMDDriftMethod
from inference_server.detectors.conformal import Conformal2
from inference_server.detectors.energydistance import EnergyDistanceDriftMethod
from inference_server.detectors.c2st import C2STDriftMethod

    

import pandas as pd
import random
import string
import time
import numpy as np

data = {}
ref_data = {}
for i in range(128):
    col = "".join(
        random.choices(
            string.ascii_letters,
            k=8
        )
    )
    data[col] = list(range(i - 100, i + 100))
    ref_data[col] = list(range(i-200, i))

data = pd.DataFrame(data)
ref_data = pd.DataFrame(data)

## BENCHMARK
n = 100

methods = [
    MMDDriftMethod(bootstrap=False),
    Conformal2(conformity_score="energy", detection_prodecure="cusum"),
    Conformal2(conformity_score="nn", detection_prodecure="sr"),
    #C2STDriftMethod(bootstrap=True),
    EnergyDistanceDriftMethod(bootstrap=True)
]

times = np.zeros((len(methods), n))


for j in range(n):
    for i, m in enumerate(methods):
        t0 = time.time()
        m(data, ref_data)
        times[i, j] = time.time() - t0


t_mu = np.mean(times[:, 1:], axis=1)
t_std = np.std(times[:, 1:], axis=1)

for m, tm, ts in zip(methods, t_mu, t_std):
    print(f"{type(m).__name__:>25}: {tm:.3f} +- {ts:.5f}")

