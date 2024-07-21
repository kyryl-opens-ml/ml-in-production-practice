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
    name=STORAGE_NAME,
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=STORAGE_NAME
    ),
)
volume_mount = k8s.V1VolumeMount(name=STORAGE_NAME, mount_path="/tmp/", sub_path=None)

with DAG(
    start_date=datetime(2021, 1, 1),
    catchup=False,
    schedule_interval=None,
    dag_id="training_dag",
) as dag:
    
    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image=DOCKER_IMAGE,
        cmds=["rm", "-rf", "/tmp/*"],
        task_id="clean_storage_before_start",
        is_delete_operator_pod=False,
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
        is_delete_operator_pod=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    train_model = KubernetesPodOperator(
        name="train_model",
        image=DOCKER_IMAGE,
        cmds=[
            "python",
            "classic_example/cli.py",
            "train",
            "tests/data/test_config.json",
        ],
        task_id="train_model",
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_model = KubernetesPodOperator(
        name="upload_model",
        image=DOCKER_IMAGE,
        cmds=[
            "python",
            "classic_example/cli.py",
            "upload-to-registry",
            "airflow-pipeline",
            "/tmp/results",
        ],
        task_id="upload_model",
        env_vars={"WANDB_PROJECT": WANDB_PROJECT, "WANDB_API_KEY": WANDB_API_KEY},
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image=DOCKER_IMAGE,
        cmds=["rm", "-rf", "/tmp/*"],
        task_id="clean_up",
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data >> train_model >> upload_model >> clean_up
