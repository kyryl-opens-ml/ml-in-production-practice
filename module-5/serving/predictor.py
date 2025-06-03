import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
import wandb
from filelock import FileLock
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger()

MODEL_ID = "truskovskiyk/ml-in-production-practice/airflow-pipeline:latest"
MODEL_PATH = "/tmp/model"
MODEL_LOCK = ".lock-file"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


class Predictor:
    def __init__(self, model_load_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: List[str]):
        text_encoded = self.tokenizer.batch_encode_plus(
            list(text), return_tensors="pt", padding=True
        )
        bert_outputs = self.model(**text_encoded).logits
        return softmax(bert_outputs, dim=-1).numpy()

    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / "model.safetensors").exists():
                load_from_registry(model_name=MODEL_ID, model_path=MODEL_PATH)
        return cls(model_load_path=MODEL_PATH)

    def run_inference_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        correct_sentence_conf = []
        for idx in tqdm(range(len(df))):
            sentence = df.iloc[idx]["sentence"]
            conf = self.predict([sentence]).flatten()[1]
            correct_sentence_conf.append(conf)
        df["correct_sentence_conf"] = correct_sentence_conf
        return df
