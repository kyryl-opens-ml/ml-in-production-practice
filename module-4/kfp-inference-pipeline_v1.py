import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar

IMAGE = "kyrylprojector/nlp-sample:latest"
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


@dsl.pipeline(name="nlp_inference_pipeline", description="nlp_inference_pipeline")
def nlp_inference_pipeline():
    load_data = dsl.ContainerOp(
        name="load_data",
        command="python nlp_sample/cli.py load-cola-data /tmp/data/".split(),
        image=IMAGE,
        file_outputs={"test": "/tmp/data/test.csv"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    load_model = dsl.ContainerOp(
        name="load_model",
        command="python nlp_sample/cli.py load-from-registry kfp-pipeline:latest /tmp/results/".split(),
        image=IMAGE,
        file_outputs={
            "config": "/tmp/results/config.json",
            "model": "/tmp/results/pytorch_model.bin",
            "tokenizer": "/tmp/results/tokenizer.json",
        },
    )
    load_model.execution_options.caching_strategy.max_cache_staleness = "P0D"

    run_inference = dsl.ContainerOp(
        name="run_inference",
        command="python nlp_sample/cli.py run-inference-on-dataframe /tmp/data/test.csv /tmp/results/ /tmp/pred.csv".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(load_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(load_model.outputs["model"], path="/tmp/results/pytorch_model.bin"),
            dsl.InputArgumentPath(load_model.outputs["tokenizer"], path="/tmp/results/tokenizer.json"),
            dsl.InputArgumentPath(load_data.outputs["test"], path="/tmp/data/test.csv"),
        ],
        file_outputs={
            "pred": "/tmp/pred.csv",
        },
    )

    env_var_project = V1EnvVar(name="WANDB_PROJECT", value=WANDB_PROJECT)
    load_model = load_model.add_env_variable(env_var_project)

    # TODO: should be a secret, but out of scope for this webinar
    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=WANDB_API_KEY)
    load_model = load_model.add_env_variable(env_var_password)


def compile_pipeline() -> str:
    path = "/tmp/nlp_inference_pipeline.yaml"
    kfp.compiler.Compiler().compile(nlp_inference_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("inference", namespace=namespace)

    print("Uploading pipeline")
    name = "nlp_inference_pipeline"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version.")
        pipeline_prev_version = client.get_pipeline(client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.id,
        )
    else:
        pipeline = client.upload_pipeline(pipeline_package_path=compile_pipeline(), pipeline_name=name)
    print(f"pipeline {pipeline.id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)
