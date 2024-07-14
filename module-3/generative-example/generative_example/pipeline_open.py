from openai import OpenAI
from random import randrange
import torch
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm
import json
from datasets import Dataset
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

import numpy as np

cache_directory = ".cache"
memory = Memory(cache_directory)
persistent_cache = memory.cache



# @persistent_cache
def get_sql(query: str, context: str, pipe) -> str:
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
    sql = output[0]['generated_text']
    match = re.search(r'\{(.*?)\}', sql, re.DOTALL)
    match.group(0)
    return json.loads(match.group(0))['sql']


def pipeline(test_json: str):


    dataset = Dataset.from_json(test_json)



    torch.random.manual_seed(0) 
    model = AutoModelForCausalLM.from_pretrained( 
        "microsoft/Phi-3-mini-128k-instruct",  
        device_map="cuda",  
        torch_dtype="auto",  
        trust_remote_code=True,  
    ) 

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    ) 

    generate_sql = []
    gt_sql = []

    for row in tqdm(dataset):
        _generate_sql = get_sql(query=row['question'], context=row['context'])
        _gt_sql = row['answer']

        generate_sql.append(_generate_sql)
        gt_sql.append(_gt_sql)

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_sql, references=gt_sql)
    print(f"results = {results}")


if __name__ == "__main__":
    pipeline()
