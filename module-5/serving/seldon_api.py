import logging
from typing import List

from serving.predictor import Predictor

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, text, features_names: List[str]):
        logger.info(text)
        results = self.predictor.predict(text)
        logger.info(results)
        return results
