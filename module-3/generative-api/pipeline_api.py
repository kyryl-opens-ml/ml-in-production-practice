import json

import evaluate
from datasets import Dataset
from joblib import Memory
from openai import OpenAI
from tqdm import tqdm
import typer
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

    return json.loads(chat_completion.choices[0].message.content)["sql"]


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
