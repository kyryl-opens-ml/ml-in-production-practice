import os

import modal
from modal import Image


app = modal.App("ml-in-production-practice-dagster-pipeline")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/dagster-pipeline:pr-14").env(env)
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
def evaluation_job(df, model_load_path):
    from text2sql_pipeline import evaluate_model
    metrics = evaluate_model(df=df, model_load_path=model_load_path)
    return metrics


@app.local_entrypoint()
def main():

    from text2sql_pipeline import _get_sql_data, AutoTokenizer, create_message_column, partial, format_dataset_chatml
    subsample = 0.1
    dataset = _get_sql_data(subsample=subsample)
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(partial(format_dataset_chatml, tokenizer=tokenizer))
    dataset_chatml_pandas = {'train': dataset_chatml['train'].to_pandas(), 'test': dataset_chatml['test'].to_pandas()}
    
    # run the function locally
    print(training_job.remote(dataset_chatml_pandas=dataset_chatml_pandas))

if __name__ == "__main__":
    main()
