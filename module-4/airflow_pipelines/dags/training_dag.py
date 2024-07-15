from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

volume = k8s.V1Volume(
    name="training-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="training-storage"),
)
volume_mount = k8s.V1VolumeMount(name="training-storage", mount_path="/tmp/", sub_path=None)

with DAG(start_date=datetime(2021, 1, 1), catchup=False, schedule_interval=None, dag_id="training_dag") as dag:
    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image="kyrylprojector/nlp-sample:latest",
        cmds=["rm", "-rf", "/tmp/*"],
        task_id="clean_storage_before_start",
        is_delete_operator_pod=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    load_data = KubernetesPodOperator(
        name="load_data",
        image="kyrylprojector/nlp-sample:latest",
        cmds=["python", "nlp_sample/cli.py", "load-cola-data", "/tmp/data/"],
        task_id="load_data",
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    train_model = KubernetesPodOperator(
        name="train_model",
        image="kyrylprojector/nlp-sample:latest",
        cmds=["python", "nlp_sample/cli.py", "train", "tests/data/test_config.json"],
        task_id="train_model",
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_model = KubernetesPodOperator(
        name="upload_model",
        image="kyrylprojector/nlp-sample:latest",
        cmds=["python", "nlp_sample/cli.py", "upload-to-registry", "airflow-pipeline", "/tmp/results"],
        task_id="upload_model",
        env_vars={"WANDB_PROJECT": "course-27-10-2023-week-3", "WANDB_API_KEY": ""},
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image="kyrylprojector/nlp-sample:latest",
        cmds=["rm", "-rf", "/tmp/*"],
        task_id="clean_up",
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data >> train_model >> upload_model >> clean_up
