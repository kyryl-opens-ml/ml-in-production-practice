import logging

import numpy as np
from transformers import pipeline

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
    sequence = sequence.tolist()

    logger.info(f"sequence = {sequence}")

    classification_result = CLASSIFIER(sequence, LABELS)
    result_labels = []
    for result in classification_result:
        logger.debug(result)
        most_probable_label = result["labels"][0]
        result_labels.append([most_probable_label])

    return {"label": np.char.encode(result_labels, "utf-8")}


def main():

    with Triton() as triton:
        logger.info("Loading BART model.")
        triton.bind(
            model_name="BART",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="label", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(max_batch_size=4),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()