import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from serving.predictor import Predictor

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

predictor = Predictor.default_from_model_registry()


@batch
def _infer_fn(text: np.ndarray):
    text = np.char.decode(text.astype("bytes"), "utf-8")
    text = text.tolist()[0]

    logger.info(f"sequence = {text}")
    results = predictor.predict(text=text)
    logger.info(f"results = {results}")
    return [results]


def main():

    with Triton() as triton:
        logger.info("Loading BART model.")
        triton.bind(
            model_name="predictor_a",
            infer_func=_infer_fn,
            inputs=[Tensor(name="text", dtype=bytes, shape=(-1,)),],
            outputs=[Tensor(name="probs", dtype=np.float32, shape=(-1,)),],            
            config=ModelConfig(max_batch_size=1),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()