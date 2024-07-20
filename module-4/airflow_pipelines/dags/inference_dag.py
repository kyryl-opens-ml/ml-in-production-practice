import os
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

DOCKER_IMAGE = "ghcr.io/kyryl-opens-ml/classic-example:main"
STORAGE_NAME = "training-storage"
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

volume = k8s.V1Volume(
    name="inference-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="inference-storage"
    ),
)
volume_mount = k8s.V1VolumeMount(
    name="inference-storage", mount_path="/tmp/", sub_path=None
)

with DAG(
    start_date=datetime(2021, 1, 1),
    catchup=False,
    schedule_interval=None,
    dag_id="inference_dag",
) as dag:
    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image=DOCKER_IMAGE,
        cmds=["rm", "-rf", "/tmp/data/*"],
        task_id="clean_storage_before_start",
        in_cluster=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    load_data = KubernetesPodOperator(
        name="load_data",
        image=DOCKER_IMAGE,
        cmds=["python", "classic_example/cli.py", "load-sst2-data", "/tmp/data/"],
        task_id="load_data",
        in_cluster=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    load_model = KubernetesPodOperator(
        name="load_model",
        image=DOCKER_IMAGE,
        cmds=[
            "python",
            "classic_example/cli.py",
            "load-from-registry",
            "airflow-pipeline:latest",
            "/tmp/results/",
        ],
        task_id="load_model",
        env_vars={"WANDB_PROJECT": WANDB_PROJECT, "WANDB_API_KEY": WANDB_API_KEY},
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    run_inference = KubernetesPodOperator(
        name="run_inference",
        image=DOCKER_IMAGE,
        cmds=[
            "python",
            "classic_example/cli.py",
            "run-inference-on-dataframe",
            "/tmp/data/test.csv",
            "/tmp/results/",
            "/tmp/pred.csv",
        ],
        task_id="run_inference",
        in_cluster=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image=DOCKER_IMAGE,
        cmds=["rm", "-rf", "/tmp/data/*"],
        task_id="clean_up",
        in_cluster=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data
    clean_storage_before_start >> load_model

    load_data >> run_inference
    load_model >> run_inference
    run_inference >> clean_up
