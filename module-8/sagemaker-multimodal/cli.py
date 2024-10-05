import typer

import boto3, json, sagemaker, time
from sagemaker import get_execution_role
import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
import tarfile 
import os
from typing import Any


class Settings(BaseSettings):
    role: str = "arn:aws:iam::469651751916:role/sagemaker-execution-role"
    model_data_url: str = "s3://sagemaker-us-east-1-469651751916/models/"
    bucket_name: str = "sagemaker-us-east-1-469651751916"
    mme_triton_image_uri: str = '785573368785.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:22.07-py3'
    model_name: str = "sagemaker-poc"
    endpoint_config_name: str = "sagemaker-poc"
    endpoint_name: str = "sagemaker-poc"

settings = Settings()
console = Console()

def create_endpoint():
    sm_client = boto3.client(service_name="sagemaker")

    container = {"Image": settings.mme_triton_image_uri, "ModelDataUrl": settings.model_data_url, "Mode": "MultiModel"}
    create_model_response = sm_client.create_model(ModelName=settings.model_name, ExecutionRoleArn=settings.role, PrimaryContainer=container)
    console.print(create_model_response)


    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=settings.endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.g5.xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": settings.model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    console.print(create_endpoint_config_response)
    create_endpoint_response = sm_client.create_endpoint(EndpointName=settings.endpoint_name, EndpointConfigName=settings.endpoint_config_name)
    console.print(create_endpoint_response)

def add_model(model_directory: str, tarball_name: str):    
    s3_key = f"models/{tarball_name}"

    with tarfile.open(tarball_name, "w:gz") as tar:
        tar.add(model_directory, arcname=os.path.basename(model_directory))
    console.print(f"Created tarball: {tarball_name}")

    s3_client = boto3.client('s3')
    s3_client.upload_file(tarball_name, settings.bucket_name, f"models/{tarball_name}")  # Use the S3 key here
    console.print(f"Uploaded model to: s3://{settings.bucket_name}/{s3_key}")
    return f"s3://{settings.bucket_name}/{s3_key}"

def _call_model(target_model: str, payload: Any):
    runtime_sm_client = boto3.client("sagemaker-runtime")
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=settings.endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(payload),
        TargetModel=target_model,
    )

    response = json.loads(response["Body"].read().decode("utf8"))
    output = response["outputs"][0]["data"]

    console.print(output)


def call_model_image(target_model: str):

    def get_sample_image():
        # Generate a random image (224x224 pixels with 3 color channels)
        img = np.random.rand(224, 224, 3).astype(np.float32)

        # Normalize the image
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3))  # Subtract mean
        img = img / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)  # Divide by std

        # Transpose the image to (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return img.tolist()
    
    pt_payload = {
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": [1, 3, 224, 224],
                "datatype": "FP32",
                "data": get_sample_image(),
            }
        ]
    }
    _call_model(target_model=target_model, payload=pt_payload)




def call_model_vector(target_model: str):
    pt_payload = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [4],
                "datatype": "FP32",
                "data": [1, 2, 3, 4],
            },
            {
                "name": "INPUT1",
                "shape": [4],
                "datatype": "FP32",
                "data": [1, 2, 3, 4],
            }            
        ]
    }
    _call_model(target_model=target_model, payload=pt_payload)

if __name__ == "__main__":
    app = typer.Typer()
    app.command()(create_endpoint)
    app.command()(add_model)
    app.command()(call_model_image)
    app.command()(call_model_vector)
    app()
