import numpy as np
import pytest
from transformers import EvalPrediction

from nlp_sample.utils import compute_metrics


@pytest.fixture()
def eval_pred() -> EvalPrediction:
    label_ids = np.ones(10)
    predictions = np.zeros((10, 2))
    predictions[:, 0] = -1
    predictions[:, 1] = 1

    _eval_pred = EvalPrediction(label_ids=label_ids, predictions=predictions)
    return _eval_pred


def test_compute_metrics(eval_pred: EvalPrediction):
    metrics = compute_metrics(eval_pred)
    assert metrics == {"f1": 1.0, "f0.5": 1.0}
