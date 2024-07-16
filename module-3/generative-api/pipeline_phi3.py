import json
import re

import evaluate
import torch
from datasets import Dataset
from joblib import Memory
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import typer

cache_directory = ".cache"
memory = Memory(cache_directory)
persistent_cache = memory.cache

def create_pipeline(phi_model_name: str = "microsoft/Phi-3-mini-128k-instruct"):
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        phi_model_name,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(phi_model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe

print("Creating pipeline")
pipe = create_pipeline()

@persistent_cache
def get_sql(query: str, context: str) -> str:
    prompt = f""""
    Write the corresponding SQL query based on user requests and database context:

    User requests: {query}
    Database context: {context}

    Please return in JSON format: {{"sql": ""}}
    """

    messages = [
        {"role": "system", "content": "You are a SQL expert."},
        {"role": "user", "content": prompt},
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    generated_text = output[0]["generated_text"]
    match = re.search(r"\{(.*?)\}", generated_text, re.DOTALL)
    matched_text = match.group(0)
    try:
        sql = json.loads(matched_text)["sql"]
    except json.JSONDecodeError:
        print(f"Cannot parse {generated_text} output")
        sql = ""
    return sql


def run_pipeline(test_json: str):
    dataset = Dataset.from_json(test_json)

    generated_sql = []
    gt_sql = []

    for row in tqdm(dataset):
        _generate_sql = get_sql(query=row["question"], context=row["context"])
        _gt_sql = row["answer"]

        generated_sql.append(_generate_sql)
        gt_sql.append(_gt_sql)

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=generated_sql, references=gt_sql)
    print(f"results = {results}")


if __name__ == "__main__":
    typer.run(run_pipeline)