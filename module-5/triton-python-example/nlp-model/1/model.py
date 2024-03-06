import os
import albumentations as A
import boto3
import numpy as np
import torch
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import requests
import json
from pathlib import Path


import json
import triton_python_backend_utils as pb_utils
import torchvision

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

MODEL_ID = "truskovskiyk/course-27-10-2023-week-3/airflow-pipeline:latest"
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
        text_encoded = self.tokenizer.batch_encode_plus(list(text), return_tensors="pt", padding=True)
        bert_outputs = self.model(**text_encoded).logits
        return softmax(bert_outputs).numpy()

    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / "pytorch_model.bin").exists():
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

    
class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "pred_boxes")
        output1_config = pb_utils.get_output_config_by_name(model_config, "scores")
        output2_config = pb_utils.get_output_config_by_name(model_config, "pred_classes")

        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])
        self.output2_dtype = pb_utils.triton_string_to_numpy(output2_config["data_type"])

        self.Predictor = Predictor.default_from_model_registry()

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "text")
            print(in_0.as_numpy())
            url = str(in_0.as_numpy()[0], encoding="utf-8")
            print(url, type(url))

            output = self.damage_segmentation_model.process_image(url=url)

            out_tensor_0 = pb_utils.Tensor("pred_boxes", output["pred_boxes"].astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("scores", output["scores"].astype(output1_dtype))
            out_tensor_2 = pb_utils.Tensor("pred_classes", output["pred_classes"].astype(output2_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
