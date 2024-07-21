import os

import modal
from modal import Image


app = modal.App("ml-in-production-practice-dagster-pipeline")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/generative-example:pr-11").env(env)
mount = modal.Mount.from_local_python_packages("dagster_pipelines", "dagster_pipelines")
timeout=10 * 60 * 60


@app.function(image=custom_image, gpu="A100", mounts=[mount], timeout=timeout)
def training_job(dataset_chatml):
    from text2sql_pipeline import train_model
    model_name, uri = train_model(dataset_chatml=dataset_chatml)
    return model_name, uri

@app.function(image=custom_image, gpu="A100", mounts=[mount], timeout=timeout)
def evaluation_job(model_name):
    pass 

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

    # run the function locally
    print(training_job.remote(dataset_chatml=dataset_chatml))

if __name__ == "__main__":
    main()
