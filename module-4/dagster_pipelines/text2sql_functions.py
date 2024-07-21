import os

import modal
from modal import Image

app = modal.App("ml-in-production-practice-dagster-pipeline")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/dagster-pipeline:main").env(env)
mount = modal.Mount.from_local_python_packages("dagster_pipelines", "dagster_pipelines")
timeout=10 * 60 * 60


@app.function(image=custom_image, gpu="a10g", mounts=[mount], timeout=timeout)
def training_job(dataset_chatml_pandas):
    from datasets import Dataset
    from text2sql_pipeline import train_model
    dataset_chatml = {'train': Dataset.from_pandas(dataset_chatml_pandas['train']), 'test': Dataset.from_pandas(dataset_chatml_pandas['test'])}
    model_name, uri = train_model(dataset_chatml=dataset_chatml)
    return model_name, uri

@app.function(image=custom_image, gpu="a10g", mounts=[mount], timeout=timeout)
def evaluation_job(df, model_name):
    from text2sql_pipeline import evaluate_model
    metrics = evaluate_model(df=df, model_name=model_name)
    return metrics

