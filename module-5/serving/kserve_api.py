from serving.predictor import Predictor
from typing import Dict
from kserve import Model, ModelServer

class CustomModel(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        self.predictor = Predictor.default_from_model_registry()
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        instances = payload["instances"]
        predictions = self.predictor.predict(instances)
        return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    model = CustomModel("custom-model")
    ModelServer().start([model])
