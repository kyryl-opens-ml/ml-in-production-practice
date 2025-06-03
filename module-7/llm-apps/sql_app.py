import json

from openai import OpenAI
import openai
import typer
from typing import Tuple
from datasets import load_dataset
import random
import agentops
from rich.console import Console
from langsmith.wrappers import wrap_openai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

console = Console()


def get_random_datapoint() -> Tuple[str, str]:
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset_size = len(dataset)

    index = random.randint(0, dataset_size - 1)
    sample = dataset[index]

    sql_context = sample["sql_context"]
    sql_prompt = sample["sql_prompt"]
    return sql_context, sql_prompt


def get_sql(query: str, context: str, client: OpenAI) -> str:
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
        model="gpt-4o-mini-2024-07-18",
        response_format={"type": "json_object"},
    )

    return json.loads(chat_completion.choices[0].message.content)["sql"]


def run_pipeline():
    sql_context, sql_prompt = get_random_datapoint()

    console.print("1. Agentops", style="bold green")
    agentops.init()
    client_agentops = openai.Client()
    result = get_sql(query=sql_prompt, context=sql_context, client=client_agentops)
    agentops.end_all_sessions()

    console.print("2. LangSmith", style="bold green")
    client_lang_smith = wrap_openai(openai.Client())
    result = get_sql(query=sql_prompt, context=sql_context, client=client_lang_smith)

    console.print("3. OpenllMetry", style="bold green")
    Traceloop.init(app_name="text2sql")

    client_traceloop = openai.Client()
    get_sql_traceloop = workflow(name="get_sql")(get_sql)
    result = get_sql_traceloop(
        query=sql_prompt, context=sql_context, client=client_traceloop
    )

    print(f"result = {result}")


if __name__ == "__main__":
    typer.run(run_pipeline)
