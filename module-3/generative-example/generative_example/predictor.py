import logging

import torch
from torch.nn.functional import softmax
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import pipeline
import evaluate
import json

logger = logging.getLogger()


class Predictor:
    def __init__(self, model_load_path: str):
        device_map = {"": 0}
        new_model = AutoPeftModelForCausalLM.from_pretrained(
            model_load_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16, #torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )
        merged_model = new_model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(model_load_path,trust_remote_code=True)
        pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
        self.pipe = pipe

    @torch.no_grad()
    def predict(self, question: str, context: str) -> str:
        pipe = self.pipe

        messages = [{
            "content": f"{context}\n Input: {question}",
            "role": "user"
        }]


        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time= 180)
        sql = outputs[0]['generated_text'][len(prompt):].strip()
        return sql


def run_inference_on_json(json_path: Path, model_load_path: Path, result_path: Path):
    df = Dataset.from_json(str(json_path)).to_pandas()
    model = Predictor(model_load_path=model_load_path)

    generated_sql = []
    for idx in tqdm(range(len(df))):
        context = df.iloc[idx]["context"]
        question = df.iloc[idx]["question"]

        sql = model.predict(question=question, context=context)
        generated_sql.append(sql)
    df["generated_sql"] = generated_sql
    df.to_csv(result_path, index=False)

def run_evaluate_on_json(json_path: Path, model_load_path: Path, result_path: Path):
    df = Dataset.from_json(str(json_path)).to_pandas()
    model = Predictor(model_load_path=model_load_path)

    generated_sql = []
    for idx in tqdm(range(len(df))):
        context = df.iloc[idx]["context"]
        question = df.iloc[idx]["question"]

        sql = model.predict(question=question, context=context)
        generated_sql.append(sql)
    
    gt_sql = df['answer'].values
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_sql, references=gt_sql)
    with open(result_path, 'w') as f:
        json.dump(results, f)


