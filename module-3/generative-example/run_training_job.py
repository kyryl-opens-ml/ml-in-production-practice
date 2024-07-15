import os

import modal
from modal import Image


app = modal.App("ml-in-production-practice")
env = {"WANDB_PROJECT": os.getenv("WANDB_PROJECT"), "WANDB_API_KEY": os.getenv("WANDB_API_KEY")}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/classic-example:pr-11").env(env)


@app.function(image=custom_image, gpu="a10g", timeout=15 * 60)
def run_training_modal():
    from generative_example.data import load_sql_data
    from generative_example.train import train
    from generative_example.utils import load_from_registry, upload_to_registry
    from generative_example.predictor import run_evaluate_on_json, run_inference_on_json

    load_sql_data(path_to_save='/tmp/data')
    train(config_path="/app/conf/example.json")
    upload_to_registry(model_name='modal-classic-example', model_path='./phi-3-mini-lora-text2sql')
    load_from_registry(model_name='modal-classic-example', model_path='./phi-3-mini-lora-text2sql-loaded-model')
    
    run_inference_on_dataframe(df_path='/tmp/data/test.csv', model_load_path='loaded-model', result_path='/tmp/inference.csv')


@app.local_entrypoint()
def main():
    print(run_training_modal.spawn())
