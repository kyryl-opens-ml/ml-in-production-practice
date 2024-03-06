import logging

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger()


class Predictor:
    def __init__(self, model_load_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        text_encoded = self.tokenizer.batch_encode_plus(list(text), return_tensors="pt", padding=True)
        bert_outputs = self.model(**text_encoded).logits
        return softmax(bert_outputs).numpy()


def run_inference_on_dataframe(df_path: Path, model_load_path: Path, result_path: Path):
    df = pd.read_csv(df_path)
    model = Predictor(model_load_path=model_load_path)

    correct_sentence_conf = []
    for idx in tqdm(range(len(df))):
        sentence = df.iloc[idx]["sentence"]
        conf = model.predict([sentence]).flatten()[1]
        correct_sentence_conf.append(conf)
    df["correct_sentence_conf"] = correct_sentence_conf
    df.to_csv(result_path, index=False)
