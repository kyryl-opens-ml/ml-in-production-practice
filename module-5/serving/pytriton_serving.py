import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from serving.predictor import Predictor

logger = logging.getLogger("pytriton_serving")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@batch
def _infer_fn(text: np.ndarray):
    print(text)
    return {"probs": np.array([[0.32, 0.312]])} 


def main():
    with Triton() as triton:
        logger.info("Loading model.")
        triton.bind(
            model_name="predictor_a",
            infer_func=_infer_fn,
            inputs=[Tensor(name="text", dtype=object, shape=(-1,))],
            outputs=[Tensor(name="probs", dtype=np.float32, shape=(-1,)),],
            config=ModelConfig(max_batch_size=4),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()