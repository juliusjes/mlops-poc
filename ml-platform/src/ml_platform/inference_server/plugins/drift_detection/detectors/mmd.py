from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    DriftMethod,
    MMD2u,
    MMD2u_bstrp,
    get_pca_df,
)
from evidently.pydantic_utils import autoregister
from sklearn.metrics import pairwise_distances, pairwise_kernels


@autoregister
class MMDDriftMethod(DriftMethod):
    class Config:
        type_alias = "evidently:drift_method:MMDDriftMethod"

    threshold: float = 0.015
    bootstrap: Optional[bool] = None
    quantile_probability: float = 0.05
    pca_components: Optional[int] = None
    n_bootstrap = 100

    def __call__(self, current_emb: pd.DataFrame, reference_emb: pd.DataFrame) -> Tuple[float, bool, str]:
        if self.pca_components:
            reference_emb, current_emb = get_pca_df(reference_emb, current_emb, self.pca_components)
        x = reference_emb
        y = current_emb
        m = len(x)
        n = len(y)

        pair_dists = pairwise_distances(
            x.sample(min(m, 1000), random_state=0),
            y.sample(min(n, 1000), random_state=0),
            metric="euclidean",
            n_jobs=1,
        )
        sigma2 = np.median(pair_dists) ** 2
        xy = np.vstack([x, y])
        K = pairwise_kernels(xy, metric="rbf", gamma=1.0 / sigma2)
        mmd2u = MMD2u(K, m, n)
        if self.bootstrap:
            pair_dists_bstrp = pairwise_distances(x.sample(min(m, 1000), random_state=0), metric="euclidean", n_jobs=1)
            sigma2_x = np.median(pair_dists_bstrp) ** 2
            gamma_x = 1.0 / sigma2_x
            K = pairwise_kernels(x, metric="rbf", gamma=gamma_x)
            x_size = max(int(m * m / (m + n)), 1)
            y_size = max(int(m * n / (m + n)), 1)
            bstrp_res = []
            for i in range(self.n_bootstrap):
                np.random.seed(i)
                x_idxs = np.random.choice(m, x_size)
                y_idxs = np.random.choice(m, y_size)
                bstrp_res.append(MMD2u_bstrp(K, x_size, y_size, x_idxs, y_idxs))

            perc = np.percentile(bstrp_res, 100 * self.quantile_probability)
            return max(mmd2u, 0), mmd2u > perc, "mmd"
        else:
            return max(mmd2u, 0), mmd2u > self.threshold, "mmd"