from openai import OpenAI
from random import randrange
import torch
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm
import json
from datasets import Dataset

import numpy as np

cache_directory = ".cache"
memory = Memory(cache_directory)
persistent_cache = memory.cache



@persistent_cache
def get_sql(query: str, context: str) -> str:
    client = OpenAI()
    prompt = f""""
    Write the corresponding SQL query based on user requests and database context:

    User requests: {query}
    Database context: {context}

    Please return in JSON format: {{"sql": ""}}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
        response_format={"type": "json_object"},
    )
    
    return json.loads(chat_completion.choices[0].message.content)['sql']


def pipeline(test_json: str):
    dataset = Dataset.from_json(test_json)

    generate_sql = []
    gt_sql = []
    for row in tqdm(dataset):
        _generate_sql = get_sql(query=query, context=context)
        _gt_sql = row['answer']

        generate_sql.append(_generate_sql)
        gt_sql.append(_gt_sql)

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_sql, references=gt_sql)
    print(f"results = {results}")


if __name__ == "__main__":
    pipeline()
