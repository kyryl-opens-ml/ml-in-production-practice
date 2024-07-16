import os

import modal
from modal import Image

app = modal.App("ml-in-production-practice")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/classic-example:pr-11").env(env)


@app.function(image=custom_image, gpu="a10g", timeout=15 * 60)
def run_classic_example():
    from pathlib import Path

    from classic_example.data import load_sst2_data
    from classic_example.predictor import run_inference_on_dataframe
    from classic_example.train import train
    from classic_example.utils import load_from_registry, upload_to_registry

    load_sst2_data(path_to_save=Path("/tmp/data/"))
    train(config_path=Path("/app/conf/example.json"))
    upload_to_registry(model_name="modal-classic-example", model_path=Path("results"))
    load_from_registry(
        model_name="modal-classic-example", model_path=Path("loaded-model")
    )
    run_inference_on_dataframe(
        df_path=Path("/tmp/data/test.csv"),
        model_load_path=Path("loaded-model"),
        result_path=Path("/tmp/data/inference.csv"),
    )


def main():
    fn = modal.Function.lookup("ml-in-production-practice", "run_classic_example")
    fn_id = fn.spawn()
    print(f"Run training object: {fn_id}")


if __name__ == "__main__":
    main()
