import logging
import time
from typing import List, Optional

import numpy as np
from sklearn.metrics import f1_score

from serving.predictor import Predictor

logger = logging.getLogger()

class Score:
    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn


class SeldonAPI:
    def __init__(self, model_id: Optional[str] = None):
        self.predictor = Predictor.default_from_model_registry(model_id=model_id)
        self._run_time = None
        self.scores = Score(tp=0, fp=0, tn=0, fn=0)

    def predict(self, text, features_names: List[str]):
        logger.info(text)

        s = time.perf_counter()
        results = self.predictor.predict(text)
        elapsed = time.perf_counter() - s
        self._run_time = elapsed

        logger.info(results)
        return results

    def metrics(self):
        return [
            {"type": "GAUGE", "key": "gauge_runtime", "value": self._run_time},
            {"type": "GAUGE", "key": f"true_pos", "value": self.scores.tp},
            {"type": "GAUGE", "key": f"true_neg", "value": self.scores.fn},
            {"type": "GAUGE", "key": f"false_pos", "value": self.scores.fn},
            {"type": "GAUGE", "key": f"false_neg", "value": self.scores.fp},
        ]

    def send_feedback(self, features, feature_names, reward, truth, routing=""):
        logger.info("features")
        logger.info(features)

        logger.info("truth")
        logger.info(truth)

        results = self.predict(features, feature_names)
        predicted = np.argmax(results, axis=1)

        if int(truth[0]) == 1:
            if int(predicted[0]) == int(truth[0]):
                self.scores.tp += 1
            else:
                self.scores.fn += 1
        else:
            if int(predicted[0]) == int(truth[0]):
                self.scores.tn += 1
            else:
                self.scores.fp += 1

        return []
