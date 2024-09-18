from pathlib import Path
import wandb
import requests
import json 


BASE_URL = "http://localhost:8000/v1"

def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


def list_of_models():
    url = f"{BASE_URL}/models"
    response = requests.get(url)
    models = response.json()
    print(json.dumps(models, indent=4))

def load_adapter(lora_name: str, lora_path: str):

    lora_name = "sql-test"
    lora_path = "data/sql-adapter/"

    url = f"{BASE_URL}/load_lora_adapter"
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path
    }
    response = requests.post(url, json=payload)
    print(response)

def unload_adapter(lora_name: str):
    url = f"{BASE_URL}/unload_lora_adapter"
    payload = {
        "lora_name": lora_name
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    print(json.dumps(result, indent=4))

def test_client(model: str, prompt: str, max_tokens: int = 7, temperature: float = 0.0):
    prompt = "test"
    max_tokens: int = 7
    temperature: float = 0.0
    # model = "microsoft/Phi-3-mini-4k-instruct"
    model = "sql-test"
    url = f"{BASE_URL}/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    print(json.dumps(completion, indent=4))

def run_inference_on_json(json_file: Path):
    url = f"{BASE_URL}/completions"
    with open(json_file, 'r') as f:
        payload = json.load(f)
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    completion = response.json()
    print(json.dumps(completion, indent=4))




def cli():
    app = typer.Typer()
    app.command()(load_from_registry)
    app.command()(list_of_models)
    app.command()(load_adapter)
    app.command()(unload_adapter)
    app.command()(test_client)
    app.command()(upload_to_registry)
    app.command()(run_inference_on_json)
    app.command()(run_evaluate_on_json)
    app()

if __name__ == "__main__":
    cli()

