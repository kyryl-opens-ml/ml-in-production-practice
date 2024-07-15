import logging
import sys
from pathlib import Path

import datasets
import transformers
import wandb


def setup_logger(logger):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "README.md")
        art.add_file(model_path / "adapter_config.json")
        art.add_file(model_path / "adapter_model.safetensors")
        art.add_file(model_path / "special_tokens_map.json")
        art.add_file(model_path / "tokenizer.json")
        art.add_file(model_path / "tokenizer_config.json")
        art.add_file(model_path / "training_args.bin")
        wandb.log_artifact(art)


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")
