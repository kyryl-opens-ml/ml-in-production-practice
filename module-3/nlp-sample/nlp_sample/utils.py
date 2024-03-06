import logging
import sys
from pathlib import Path
from typing import Dict

import datasets
import numpy as np
import transformers
import wandb
from sklearn.metrics import f1_score, fbeta_score
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    preds = p.predictions
    preds = np.argmax(preds, axis=1)
    return {
        "f1": f1_score(y_true=p.label_ids, y_pred=preds),
        "f0.5": fbeta_score(y_true=p.label_ids, y_pred=preds, beta=0.5),
    }


def preprocess_function_examples(examples, tokenizer, padding, max_seq_length, label_to_id):
    sentence1_key = "sentence"
    args = (examples[sentence1_key],)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[label] if label != -1 else -1) for label in examples["label"]]
    return result


def setup_logger(logger):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "config.json")
        art.add_file(model_path / "model.safetensors")
        art.add_file(model_path / "tokenizer.json")
        art.add_file(model_path / "tokenizer_config.json")
        art.add_file(model_path / "special_tokens_map.json")
        art.add_file(model_path / "README.md")
        wandb.log_artifact(art)


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")
