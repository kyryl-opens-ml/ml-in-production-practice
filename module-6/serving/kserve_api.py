import argparse
import logging
from typing import Dict

import kserve

from serving.predictor import Predictor

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class CustomModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.predictor = None
        self.load()

    def load(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        logging.info(f"Received inputs {payload}")
        text = payload["instances"]
        logging.info(f"Received text {text}, {type(text)}")
        result = self.predictor.predict(text=text)
        logging.info(f"Resutls {result}, {type(result)}")
        return {"predictions": result.tolist()}


DEFAULT_MODEL_NAME = "kserve-custom"
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="The name that the model is served under.")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    custom_model = CustomModel(args.model_name)
    server = kserve.ModelServer()
    server.start(models=[custom_model])
