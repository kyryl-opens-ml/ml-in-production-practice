import logging

import numpy as np
from transformers import pipeline

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from serving.predictor import Predictor

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

predictor = Predictor.default_from_model_registry()

# Labels pre-cached on server side
LABELS = [
    "travel",
    "cooking",
    "dancing",
    "sport",
    "music",
    "entertainment",
    "festival",
    "movie",
    "literature",
]


@batch
def _infer_fn(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")
    sequence = sequence.tolist()[0]

    logger.info(f"sequence = {sequence}")
    results = predictor.predict(text=sequence)
    logger.info(f"results = {results}")

    result_labels = ['travel' for _ in range(len(sequence))]
    return {"label": np.char.encode(result_labels, "utf-8")}


def main():

    with Triton() as triton:
        logger.info("Loading BART model.")
        triton.bind(
            model_name="BART",
            infer_func=_infer_fn,
            inputs=[Tensor(name="sequence", dtype=bytes, shape=(-1,)),],
            outputs=[Tensor(name="label", dtype=bytes, shape=(1,)),],
            config=ModelConfig(max_batch_size=1),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()