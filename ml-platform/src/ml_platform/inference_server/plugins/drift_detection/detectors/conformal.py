
import time
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    DriftMethod,
    get_pca_df,
)
from evidently.pydantic_utils import autoregister

#from inference_server.detectors.energydistance import EnergyDistanceDriftMethod
from pydantic import PrivateAttr

#from matplotlib.pyplot import ecdf
from scipy.spatial.distance import cdist
from scipy.stats import kstest
from sklearn.neighbors import NearestNeighbors


def conformal_pvalues(x, x_cal):
    pvals = np.empty(len(x))
    m = len(x_cal)

    for i, xi in enumerate(x):
        greater = np.sum(x_cal > xi)
        equal = np.sum(x_cal == xi)
        u = np.random.rand()

        pvals[i] = (greater + u * (equal + 1)) / (m + 1)

    return pvals



def exp_betting_martingale(current_p, return_vector=False):
    eps = 1e-12
    p = np.clip(current_p, eps, 1.0)

    epsilon = 0.5

    betting = epsilon * p ** (epsilon - 1)
    if return_vector:
        return np.log(betting)
    return np.sum(np.log(betting))


def energy_distance(X, Y):
    XY = cdist(X, Y, metric="euclidean")
    XX = cdist(X, X, metric="euclidean")

    # Average distance from each x to Y
    cross = 2 * XY.mean(axis=1)

    # Average distance from each x to the other samples in X
    np.fill_diagonal(XX, np.nan)
    within = np.nanmean(XX, axis=1)

    return cross - within

@autoregister
class Conformal(DriftMethod):
    class Config:
        type_alias = "evidently:drift_method:Conformal"

    quantile_probability: float = 0.95 # for bootstrapping
    threshold: float = 30 # arbitrary
    nn_n_neighbors: int = 5
    bootstrap: Optional[bool] = None
    n_bootstrap: int = 10
    pca_components: Optional[int] = None
    nn_n_jobs: int = 1 # multithreading actually decreases performance in this case so 1 is fine

    _nn: NearestNeighbors = PrivateAttr()
    # _scaler: StandardScaler = PrivateAttr()
    _x_cal: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._nn = NearestNeighbors(
            n_neighbors=self.nn_n_neighbors,
            n_jobs=self.nn_n_jobs,
        )

    
    def __call__( self, current_emb: pd.DataFrame, reference_emb: pd.DataFrame) -> Tuple[float, bool, str]:

        if self.pca_components:
            reference_emb, current_emb = get_pca_df( reference_emb, current_emb, self.pca_components)

        m = len(reference_emb) # TODO: fix for sample sizes smaller than self.nn_n_neighbors

        # determine threshold
        if self.bootstrap:
            bootstrap_stats = np.empty(self.n_bootstrap)

            for i in range(self.n_bootstrap):
                rng = np.random.default_rng(i)

                perm = rng.permutation(m)

                cal_idx = perm[:m//2]
                cur_idx = perm[m//2:]

                calibration = reference_emb.iloc[cal_idx]
                pseudo_current = reference_emb.iloc[cur_idx]

                nn = self._nn

                nn.fit(calibration)

                # Calibration scores
                distances, _ = nn.kneighbors(calibration)
                x_cal = distances[:, 1:].mean(axis=1)

                # Pseudo-current scores
                distances, _ = nn.kneighbors(pseudo_current)
                x = distances[:, 1:].mean(axis=1)

                pvals = conformal_pvalues(x, x_cal)
                stat = exp_betting_martingale(pvals)

                bootstrap_stats[i] = stat

            perc = float(np.percentile(bootstrap_stats, 100 * self.quantile_probability))
        else:
            perc = self.threshold

        # compare
        self._nn.fit(reference_emb)
        distances, _ = self._nn.kneighbors()
        x_cal = distances.mean(axis=1)

        distances, _ = self._nn.kneighbors(current_emb)
        x = distances.mean(axis=1)
        current_pvalues = conformal_pvalues(x, x_cal)
        
        bmg = exp_betting_martingale(current_pvalues)

        return bmg, bmg >= perc, "conformal"

    
class ConformityScore:
    def __init__(self, method: str, **params):
        self.method = method
        match method:
            case "nn":
                self.nn = NearestNeighbors(n_neighbors=5, n_jobs=1, **params)
            case "energy":
                pass
            case _:
                pass

    def __call__(self, reference, current) -> tuple:
        if self.method == "nn":
            self.nn.fit(reference)
            d_cal, _ = self.nn.kneighbors()
            x_cal = d_cal.mean(axis=1)
            d_cur, _ = self.nn.kneighbors(current)
            x = d_cur.mean(axis=1)
            return x_cal, x

        elif self.method == "energy":
            return energy_distance(reference, reference), energy_distance(reference, current)


class TestFunction:
    def __init__(self, method, **params):
        self.method = method

    def __call__(self, p_values, return_vector=False) -> float | np.ndarray:
        if self.method == "martingale":
            return exp_betting_martingale(p_values, return_vector=return_vector)
        if self.method == "ks": # TODO: might not work as intended
            return kstest(p_values, "uniform", keepdims=return_vector).statistic
        

@autoregister
class Conformal2(DriftMethod):
    """
    Conformal martingale drift detector.

    detection_prodecure="ville":
        Anytime-valid conformal betting martingale.

    detection_prodecure="cusum":
        CUSUM statistic driven by conformal betting increments.
        Threshold is bootstrap-calibrated from the reference window.

    detection_prodecure="sr":
        SR statistic driven by conformal betting increments.
        Threshold is bootstrap-calibrated from the reference window.
    """

    class Config:
        type_alias = "evidently:drift_method:Conformal2"

    quantile_probability: float = 0.95
    detection_prodecure: Literal["ville", "cusum", "sr"] = "cusum"

    conformity_score: Literal["nn", "energy"] = "nn"
    conformity_params: dict = {}
    _conformity_func: ConformityScore = PrivateAttr()

    test_function: Literal["martingale", "ks"] = "martingale"
    test_function_params: dict = {}
    _test_function_func: TestFunction = PrivateAttr()

    alpha: float = 0.05
    bootstrap: Optional[bool] = None
    n_bootstrap: int = 100
    pca_components: Optional[int] = None

    _previous: pd.DataFrame = PrivateAttr()
    _logM: float = PrivateAttr()

    _cusum_log: float = PrivateAttr()
    _cusum_threshold_log: Optional[float] = PrivateAttr()
    _cusum_calibrated: bool = PrivateAttr()

    _sr_log: float = PrivateAttr()
    _sr_threshold_log: Optional[float] = PrivateAttr()
    _sr_calibrated: bool = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        self._conformity_func = ConformityScore(self.conformity_score, **self.conformity_params)
        self._test_function_func = TestFunction(self.test_function, **self.test_function_params)
        self._previous = pd.DataFrame([])
        self._logM = 0.0

        self._cusum_log = 0.0
        self._cusum_threshold_log = None
        self._cusum_calibrated = False

        self._sr_log = 0.0
        self._sr_threshold_log = None
        self._sr_calibrated = False

    def __call__(self, current_emb:pd.DataFrame, reference_emb: pd.DataFrame):

        if self.pca_components:
            reference_emb, current_emb = get_pca_df( reference_emb, current_emb, self.pca_components, )

        new_emb = self._get_new_embeddings(current_emb)

        self._previous = current_emb.copy()

        if len(new_emb) == 0:
            stat = self._current_statistic()
            threshold = self._threshold()
            return stat, threshold < stat, "conformal2"

        if self.detection_prodecure == "cusum" and not self._cusum_calibrated:
            self._calibrate_cusum(reference_emb)

        if self.detection_prodecure == "sr" and not self._sr_calibrated:
            self._calibrate_sr(reference_emb)

        return_vector = self.detection_prodecure == "sr"
        increment = self._compute_increment(reference_emb, new_emb, return_vector=return_vector)

        stat = self._update_statistic(increment)

        threshold = self._threshold()

        return stat, threshold < stat, "conformal2"


    def _get_new_embeddings(self, current_emb):
        if len(self._previous) == 0:
            return current_emb

        comb = pd.concat([current_emb, self._previous])
        return comb.drop_duplicates(keep=False)


    def _compute_increment(self, reference_emb, current_emb, return_vector=False):
        x_cal, x = self._conformity_func(reference_emb, current_emb)
        pvals = conformal_pvalues(x, x_cal)
        return self._test_function_func(pvals, return_vector=return_vector) 


    def _update_statistic(self, increment):

        if self.detection_prodecure == "ville":
            self._logM += increment
            return self._logM

        elif self.detection_prodecure == "cusum":
            self._cusum_log = max(0.0, self._cusum_log + increment)
            return self._cusum_log

        elif self.detection_prodecure == "sr":
            for logb in increment:
                self._sr_log = ( np.logaddexp(0.0, self._sr_log) + logb )
            return self._sr_log


    def _current_statistic(self):
        if self.detection_prodecure == "ville":
            return self._logM

        elif self.detection_prodecure == "cusum":
            return self._cusum_log

        elif self.detection_prodecure == "sr":
            return self._sr_log

        return 0.0


    def _threshold(self):
        if self.detection_prodecure == "ville":
            return np.log(1 / self.alpha)

        elif self.detection_prodecure == "cusum":
            return self._cusum_threshold_log

        elif self.detection_prodecure == "sr":
            return self._sr_threshold_log

        return False

    def _calibrate_cusum(self, reference_emb):
        maxima = []
        for _ in range(self.n_bootstrap):
            ref_boot = reference_emb.sample( len(reference_emb), replace=True).reset_index(drop=True)
            cur_boot = reference_emb.sample( len(reference_emb), replace=True).reset_index(drop=True)

            log_betting = self._compute_increment(ref_boot, cur_boot, return_vector=True)

            cs = np.cumsum(log_betting)
            mins = np.minimum.accumulate(np.concatenate(([0.0], cs[:-1])))
            cusum = cs - mins

            max_cusum = np.max(cusum)
            maxima.append(max_cusum)

        self._cusum_threshold_log = float(np.quantile( maxima, self.quantile_probability))
        self._cusum_calibrated = True


    def _calibrate_sr(self, reference_emb):
        maxima = []
        for _ in range(self.n_bootstrap):

            ref_boot = reference_emb.sample( len(reference_emb), replace=True, ).reset_index(drop=True)
            cur_boot = reference_emb.sample( len(reference_emb), replace=True, ).reset_index(drop=True)

            log_betting = self._compute_increment(ref_boot, cur_boot, return_vector=True)
            # inc is a vector of martingale values
            logR = -np.inf
            max_R = -np.inf

            for logb in log_betting:
                logR = np.logaddexp(0.0, logR) + logb
                max_R = max(max_R, logR)

            maxima.append(max_R)

        self._sr_threshold_log = float( np.quantile(maxima, self.quantile_probability) )
        self._sr_calibrated = True


    def _reset(self):
        self._previous = pd.DataFrame([])
        self._logM = 0.0

        self._cusum_log = 0.0
        self._cusum_threshold_log = None
        self._cusum_calibrated = False

        self._sr_log = 0.0
        self._sr_threshold_log = None
        self._sr_calibrated = False




if __name__ == "__main__":

    #print(exp_betting_martingale(np.random.uniform(0,1, size=1000)))
        

  
    #import matplotlib.pyplot as plt
    #from inference_server.detectors.evd import MMDDriftMethod
    # # c20 = Conformal2(detection_prodecure="cusum")
    # # c21 = Conformal2(detection_procedure="ville")
    # # c22 = Conformal(bootstrap=False)

    D = 256

    size = (100, D)
    N = 10
    x1 = pd.DataFrame(np.random.normal(0, 1, size=(100, D)))

    #detectors = [c20, c21, c22]
    detectors = [
        #MMDDriftMethod(bootstrap=False),
        #Conformal2(conformity_score="nn", test_function="martingale"),
        Conformal2(conformity_score="energy", test_function="martingale", detection_procedure="cusum"),
        #EnergyDistanceDriftMethod(bootstrap=True),
        Conformal2(conformity_score="energy", detection_procedure="sr")
        ]
    mus = np.zeros((N))
    stats = np.zeros((4, N))
    detections = np.zeros((4, N))

    mus_X = np.linspace(0, 5, N) / np.sqrt(D)  # np.random.normal(0, 0.5, size=(N))

    for i, mu in enumerate(mus_X):
        
        mus[i] = mu
        for n in range(len(detectors)):
            con = detectors[n]
            for j in range(10):
                x2 = pd.DataFrame(np.random.normal(mu, 1., size=size))
                stat, is_drift, name = con(x2, x1)
            #print(stat)
            stats[n, i] = stat
            
            if is_drift :
                detections[n, i] = stat
                if getattr(con, "_reset", None) is not None:
                    con._reset()
                #x1 = x2

    colors_s = ["red", "green", "blue", "black"]
    colors_d = ["pink", "lime", "lightblue", "gray"]
    names = ["mmd", "con_energy_m_cusum", "energy", "con_energy_m_sr"]
    mus = np.arange(0, len(mus_X))

    row_min = stats.min(axis=1, keepdims=True)
    row_max = stats.max(axis=1, keepdims=True)

    stats_norm = np.divide(
        stats - row_min,
        row_max - row_min,
        out=np.zeros_like(stats, dtype=float),
        where=(row_max - row_min) != 0,
    )

    detections_norm = np.divide(
        detections - row_min,
        row_max - row_min,
        out=np.zeros_like(detections, dtype=float),
        where=(row_max - row_min) != 0,
    )

    # for n in range(len(detectors)):
    #     plt.plot(mus_X, stats_norm[n, :], color=colors_s[n], label=names[n])
    #     plt.scatter(mus_X[detections[n,:] > 0], detections_norm[n, detections[n, :] > 0], color=colors_d[n])
    # plt.legend()
    # plt.show()