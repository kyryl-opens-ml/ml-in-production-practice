from openai import OpenAI
from random import randrange
import torch
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm
import json
from datasets import load_metric
import numpy as np

cache_directory = ".cache"
memory = Memory(cache_directory)
persistent_cache = memory.cache

rouge_metric = load_metric("rouge", trust_remote_code=True)


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

def calculate_rogue(row):

    query = row['question']
    context = row['context']

    response = get_sql(query=query, context=context)
    result = rouge_metric.compute(predictions=[response], references=[row['answer']], use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result['response']=response
    return result

def pipeline():
    dataset_name = 'b-mc2/sql-create-context'
    dataset_split = 'train'

    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset = dataset.shuffle(seed=1234).select(range(50))
    metricas = []
    for row in tqdm(dataset):
        metricas.append(calculate_rogue(row=row))


    print("Rouge 1 Mean: ",np.mean([x['rouge1'] for x in metricas]))
    print("Rouge 2 Mean: ",np.mean([x['rouge2'] for x in metricas]))
    print("Rouge L Mean: ",np.mean([x['rougeL'] for x in metricas]))
    print("Rouge Lsum Mean: ",np.mean([x['rougeLsum'] for x in metricas]))


if __name__ == "__main__":
    pipeline()
