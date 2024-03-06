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
def load_data(test_data: Output[Dataset]):
    from nlp_sample.data import load_cola_data
    from pathlib import Path
    import shutil

    load_cola_data(Path("/app/data"))

    shutil.move(Path("/app/data") / "test.csv", test_data.path)


@dsl.component(base_image=IMAGE)
def load_model(
    config: Output[Artifact],
    model: Output[Model],
    tokenizer: Output[Artifact],
    tokenizer_config: Output[Artifact],
    model_card: Output[Artifact],
    special_tokens_map: Output[Artifact],
):
    from nlp_sample.utils import load_from_registry
    from pathlib import Path
    import shutil

    model_path = Path("/tmp/model")
    model_path.mkdir(exist_ok=True)
    load_from_registry(model_name="kfp-pipeline:latest", model_path=model_path)

    shutil.move(model_path / "config.json", config.path)
    shutil.move(model_path / "model.safetensors", model.path)
    shutil.move(model_path / "tokenizer.json", tokenizer.path)
    shutil.move(model_path / "tokenizer_config.json", tokenizer_config.path)
    shutil.move(model_path / "special_tokens_map.json", model_card.path)
    shutil.move(model_path / "README.md", special_tokens_map.path)


@dsl.component(base_image=IMAGE)
def run_inference(
    config: Input[Artifact],
    model: Input[Model],
    tokenizer: Input[Artifact],
    tokenizer_config: Input[Artifact],
    model_card: Input[Artifact],
    special_tokens_map: Input[Artifact],
    test_data: Input[Dataset],
    pred: Output[Dataset],
):
    from nlp_sample.predictor import run_inference_on_dataframe
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

    run_inference_on_dataframe(df_path=test_data.path, model_load_path=model_path, result_path=pred.path)


@dsl.pipeline
def inference_pipeline():
    load_data_task = load_data()

    load_model_task = load_model()
    load_model_task = load_model_task.set_env_variable(name="WANDB_PROJECT", value=WANDB_PROJECT)
    load_model_task = load_model_task.set_env_variable(name="WANDB_API_KEY", value=WANDB_API_KEY)

    run_inference_task = run_inference(
        config=load_model_task.outputs["config"],
        model=load_model_task.outputs["model"],
        tokenizer=load_model_task.outputs["tokenizer"],
        tokenizer_config=load_model_task.outputs["tokenizer_config"],
        model_card=load_model_task.outputs["model_card"],
        special_tokens_map=load_model_task.outputs["special_tokens_map"],
        test_data=load_data_task.outputs["test_data"],
    )


def compile_pipeline() -> str:
    path = "/tmp/nlp_inference_pipeline.yaml"
    kfp.compiler.Compiler().compile(inference_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("inference", namespace=namespace)

    print("Uploading pipeline")
    name = "nlp-sample-inference"
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
