import os
import uuid
from typing import Optional
from kfp.client import Client
import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Artifact
from typing import NamedTuple


IMAGE = "kyrylprojector/nlp-sample:latest"
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


@dsl.component(base_image=IMAGE)
def load_data(train_data: Output[Dataset], val_data: Output[Dataset], test_data: Output[Dataset]):
    from nlp_sample.data import load_cola_data
    from pathlib import Path
    import shutil

    load_cola_data(Path("/app/data"))

    shutil.move(Path("/app/data") / "train.csv", train_data.path)
    shutil.move(Path("/app/data") / "val.csv", val_data.path)
    shutil.move(Path("/app/data") / "test.csv", test_data.path)


@dsl.component(base_image=IMAGE)
def train_model(
    config: Output[Artifact],
    model: Output[Model],
    tokenizer: Output[Artifact],
    tokenizer_config: Output[Artifact],
    model_card: Output[Artifact],
    special_tokens_map: Output[Artifact],
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    test_data: Input[Dataset],
):
    from nlp_sample.train import train
    from pathlib import Path
    import shutil

    Path("/tmp/data").mkdir(exist_ok=True)
    shutil.copy(train_data.path, Path("/tmp/data") / "train.csv")
    shutil.copy(val_data.path, Path("/tmp/data") / "val.csv")
    shutil.copy(test_data.path, Path("/tmp/data") / "test.csv")

    train(config_path=Path("tests/data/test_config.json"))

    shutil.move("/tmp/results/config.json", config.path)
    shutil.move("/tmp/results/model.safetensors", model.path)
    shutil.move("/tmp/results/tokenizer.json", tokenizer.path)
    shutil.move("/tmp/results/tokenizer_config.json", tokenizer_config.path)
    shutil.move("/tmp/results/special_tokens_map.json", model_card.path)
    shutil.move("/tmp/results/README.md", special_tokens_map.path)


@dsl.component(base_image=IMAGE)
def upload_model(
    config: Input[Artifact],
    model: Input[Model],
    tokenizer: Input[Artifact],
    tokenizer_config: Input[Artifact],
    model_card: Input[Artifact],
    special_tokens_map: Input[Artifact],
):
    from nlp_sample.utils import upload_to_registry
    from pathlib import Path
    import shutil

    model_path = Path("/tmp/model")
    model_path.mkdir(exist_ok=True)
    shutil.copy(config.path, model_path / "config.json")
    shutil.copy(model.path, model_path / "model.safetensors")
    shutil.copy(tokenizer.path, model_path / "tokenizer.json")
    shutil.copy(tokenizer_config.path, model_path / "tokenizer_config.json")
    shutil.copy(special_tokens_map.path, model_path / "special_tokens_map.json")
    shutil.copy(model_card.path, model_path / "README.md")

    upload_to_registry(model_name="kfp-pipeline", model_path=model_path)


@dsl.pipeline
def training_pipeline():
    load_data_task = load_data()

    train_model_task = train_model(
        train_data=load_data_task.outputs["train_data"],
        val_data=load_data_task.outputs["val_data"],
        test_data=load_data_task.outputs["test_data"],
    )
    train_model_task = train_model_task.set_env_variable(name="WANDB_PROJECT", value=WANDB_PROJECT)
    train_model_task = train_model_task.set_env_variable(name="WANDB_API_KEY", value=WANDB_API_KEY)

    upload_model_task = upload_model(
        config=train_model_task.outputs["config"],
        model=train_model_task.outputs["model"],
        tokenizer=train_model_task.outputs["tokenizer"],
        tokenizer_config=train_model_task.outputs["tokenizer_config"],
        model_card=train_model_task.outputs["model_card"],
        special_tokens_map=train_model_task.outputs["special_tokens_map"],
    )
    upload_model_task = upload_model_task.set_env_variable(name="WANDB_PROJECT", value=WANDB_PROJECT)
    upload_model_task = upload_model_task.set_env_variable(name="WANDB_API_KEY", value=WANDB_API_KEY)


def compile_pipeline() -> str:
    path = "/tmp/nlp_traininig_pipeline.yaml"
    kfp.compiler.Compiler().compile(training_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline")
    name = "nlp-sample-training"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version.")
        pipeline_prev_version = client.get_pipeline(client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.pipeline_id,
        )
    else:
        pipeline = client.upload_pipeline(pipeline_package_path=compile_pipeline(), pipeline_name=name)
    print(f"pipeline {pipeline.pipeline_id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)
