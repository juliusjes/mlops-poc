
import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
#from data_processing.data_processing import generate_simple_data
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    DriftMethod,
    get_pca_df,
)
from evidently.pydantic_utils import autoregister
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def energy_distance(X, Y):
    """
    e-statistic
    """
    # logger.debug(f"X type: {type(X)}")
    # logger.debug(f"X: {X.dtypes}")
    # logger.debug(f"Y: {Y.dtypes}")
    # logger.debug(f"X[0]: {X.iloc[0]}")

    XX = cdist(X, X, metric="euclidean")
    YY = cdist(Y, Y, metric="euclidean")
    XY = cdist(X, Y, metric="euclidean")

    return (
        2 * XY.mean()
        - XX.mean()
        - YY.mean()
    )

@autoregister
class EnergyDistanceDriftMethod(DriftMethod):
    class Config:
        type_alias = "evidently:drift_method:EnergyDistanceDriftMethod"

    threshold: float = 0.2
    bootstrap: Optional[bool] = None
    n_bootstrap: int = 10
    quantile_probability: float = 0.95
    pca_components: Optional[int] = None

    def __call__(self, current_emb: pd.DataFrame, reference_emb: pd.DataFrame) -> Tuple[float, bool, str]:
        if self.pca_components:
            reference_emb, current_emb = get_pca_df(reference_emb, current_emb, self.pca_components)
        res = energy_distance(reference_emb, current_emb)

        if self.bootstrap:
            m = len(reference_emb)
            null = []

            for i in range(self.n_bootstrap):
                rng = np.random.default_rng(i)

                perm = rng.permutation(m)

                cal_idx = perm[:m//2]
                cur_idx = perm[m//2:]

                Xp = reference_emb.iloc[cal_idx]
                Yp = reference_emb.iloc[cur_idx]

                null.append( energy_distance(Xp, Yp) ) # TODO: the pairwise distances can be precomputed and indexed into -> speedup

            perc = np.quantile( null, self.quantile_probability )
        else:
            perc = self.threshold
             
        
        return res, res > perc, "edistance"
    

if __name__ == "__main__":
    pass
    # df_ref = generate_simple_data(100, dim=128, n_classes=2)
    # df_cur = generate_simple_data(100, dim=128)


    # X_ref = np.stack(df_ref.loc[df_ref["y"] > -1, "X"].to_list())
    # X_cur = np.stack(df_cur.loc[df_cur["y"] > -1, "X"].to_list())


    # edm = EnergyDistanceDriftMethod()

    # res, det, name = edm(pd.DataFrame(X_ref), pd.DataFrame(X_cur))

    # print(res, det)