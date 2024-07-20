import os

import modal
from modal import Image

app = modal.App("ml-in-production-practice")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/generative-example:pr-11").env(env)


@app.function(image=custom_image, gpu="A100", timeout=10 * 60 * 60)
def run_generative_example():
    from pathlib import Path

    from generative_example.data import load_sql_data
    from generative_example.predictor import run_evaluate_on_json
    from generative_example.train import train
    from generative_example.utils import load_from_registry, upload_to_registry

    load_sql_data(path_to_save=Path("/tmp/data"))
    train(config_path=Path("/app/conf/example-modal.json"))
    upload_to_registry(model_name="modal_generative_example", model_path=Path("/tmp/phi-3-mini-lora-text2sql"))
    load_from_registry(model_name="modal_generative_example:latest", model_path=Path("/tmp/loaded-model"))
    run_evaluate_on_json(json_path=Path("/tmp/data/test.json"), model_load_path=Path("/tmp/loaded-model"), result_path=Path("/tmp/data/results.json"))


def main():
    fn = modal.Function.lookup("ml-in-production-practice", "run_generative_example")
    fn_id = fn.spawn()
    print(f"Run training object: {fn_id}")


if __name__ == "__main__":
    main()
