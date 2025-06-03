from pathlib import Path
import wandb
import requests
import json
import typer
from rich import print
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
EXAMPLE_CONTEXT = "CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');"
EXAMPLE_QUERY = "What is the total volume of timber sold by each salesperson, sorted by salesperson?"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


def list_of_models(url: str = DEFAULT_BASE_URL):
    url = f"{url}/models"
    response = requests.get(url)
    models = response.json()
    print(json.dumps(models, indent=4))


def load_adapter(lora_name: str, lora_path: str, url: str = DEFAULT_BASE_URL):
    url = f"{url}/load_lora_adapter"
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    response = requests.post(url, json=payload)
    print(response)


def unload_adapter(lora_name: str, url: str = DEFAULT_BASE_URL):
    url = f"{url}/unload_lora_adapter"
    payload = {"lora_name": lora_name}
    response = requests.post(url, json=payload)
    result = response.json()
    print(json.dumps(result, indent=4))


def test_client(
    model: str,
    context: str = EXAMPLE_CONTEXT,
    query: str = EXAMPLE_QUERY,
    url: str = DEFAULT_BASE_URL,
):
    client = OpenAI(base_url=url, api_key="any-api-key")
    messages = [{"content": f"{context}\n Input: {query}", "role": "user"}]
    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)


def cli():
    app = typer.Typer()
    app.command()(load_from_registry)
    app.command()(list_of_models)
    app.command()(load_adapter)
    app.command()(unload_adapter)
    app.command()(test_client)
    app()


if __name__ == "__main__":
    cli()
