from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    DriftMethod,
    calc_roc_auc_random,
    get_pca_df,
    roc_auc_score,
)
from evidently.pydantic_utils import autoregister
from pydantic import PrivateAttr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#from xgboost import XGBClassifier


@autoregister
class C2STDriftMethod(DriftMethod):
    class Config:
        type_alias = "evidently:drift_method:C2STDriftMethod"

    threshold: float = 0.55
    bootstrap: Optional[bool] = None
    quantile_probability: float = 0.95
    pca_components: Optional[int] = None

    _classifier: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # self._classifier = XGBClassifier(
        #         n_estimators=10,
        #         max_depth=2,
        #         learning_rate=0.2,
        #         eval_metric="logloss"
        #     )
        self._classifier = LogisticRegression()

    def __call__(self, current_emb: pd.DataFrame, reference_emb: pd.DataFrame) -> Tuple[float, bool, str]:
        if self.pca_components:
            reference_emb, current_emb = get_pca_df(reference_emb, current_emb, self.pca_components)
        X = pd.concat([reference_emb, current_emb], ignore_index=True)
        y = np.concatenate([ np.ones(len(reference_emb), dtype=int), np.zeros(len(current_emb), dtype=int), ])

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42, shuffle=True )

        self._classifier.fit(X_train, y_train)
        y_pred_proba = self._classifier.predict_proba(X_test)[:, 1]
        roc_aucV = roc_auc_score(y_test, y_pred_proba)

        if self.bootstrap:
            roc_auc_values = [calc_roc_auc_random(y_test, i) for i in range(100)]
            rand_roc_aucT = np.percentile(roc_auc_values, 100 * self.quantile_probability)
            return roc_aucV, roc_aucV > rand_roc_aucT, "model"
        
        return roc_aucV, roc_aucV > self.threshold, "model"