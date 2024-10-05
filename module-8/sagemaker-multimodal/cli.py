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
class Settings(BaseSettings):
    role: str = "arn:aws:iam::469651751916:role/sagemaker-execution-role"
    model_data_url: str = "s3://sagemaker-us-east-1-469651751916/models/"
    mme_triton_image_uri: str = '785573368785.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:22.07-py3'
    model_name: str = "sagemaker-poc"
    endpoint_config_name: str = "sagemaker-poc"
    endpoint_name: str = "sagemaker-poc"

settings = Settings()


def create_endpoint():
    sm_client = boto3.client(service_name="sagemaker")
    console = Console()

    container = {"Image": settings.mme_triton_image_uri, "ModelDataUrl": settings.model_data_url, "Mode": "MultiModel"}
    create_model_response = sm_client.create_model(ModelName=settings.model_name, ExecutionRoleArn=settings.role, PrimaryContainer=container)
    print(create_model_response, type(create_model_response))
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

# def add_model(bucket_name: str, model_directory: str, tarball_name: str):
def add_model():    
    bucket_name = "sagemaker-us-east-1-469651751916"  # No trailing slash
    model_directory = "./sagemaker-multimodal/model_registry/triton-serve-pt/resnet"  # The directory to be compressed
    tarball_name = "resnet_pt_v0.tar.gz"  # Name of the tarball
    s3_key = f"models/{tarball_name}"  # Define the key for the S3 object

    # Step 1: Create a tarball
    tarball_path = tarball_name  # Name of the tarball

    # Create the tarball
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(model_directory, arcname=os.path.basename(model_directory))
    print(f"Created tarball: {tarball_path}")

    # Step 2: Upload to S3 using Boto3 client
    s3_client = boto3.client('s3')
    s3_client.upload_file(tarball_path, bucket_name, s3_key)  # Use the S3 key here
    print(f"Uploaded model to: s3://{bucket_name}/{s3_key}")

    return f"s3://{bucket_name}/{s3_key}"

def add_model_2():    
    bucket_name = "sagemaker-us-east-1-469651751916"  # No trailing slash
    model_directory = "./python_backend/examples/add_sub/"  # The directory to be compressed
    tarball_name = "add_sub_v1.tar.gz"  # Name of the tarball
    s3_key = f"models/{tarball_name}"  # Define the key for the S3 object

    # Step 1: Create a tarball
    tarball_path = tarball_name  # Name of the tarball

    # Create the tarball
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(model_directory, arcname=os.path.basename(model_directory))
    print(f"Created tarball: {tarball_path}")

    # Step 2: Upload to S3 using Boto3 client
    s3_client = boto3.client('s3')
    s3_client.upload_file(tarball_path, bucket_name, s3_key)  # Use the S3 key here
    print(f"Uploaded model to: s3://{bucket_name}/{s3_key}")

    return f"s3://{bucket_name}/{s3_key}"

def call_model():

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

    endpoint_name: str = "sagemaker-poc"

    runtime_sm_client = boto3.client("sagemaker-runtime")
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(pt_payload),
        TargetModel="resnet_pt_v0.tar.gz",
    )

    response = json.loads(response["Body"].read().decode("utf8"))
    output = response["outputs"][0]["data"]

    print(output)



def call_model_2():

    
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

    endpoint_name: str = "sagemaker-poc"

    runtime_sm_client = boto3.client("sagemaker-runtime")
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(pt_payload),
        TargetModel="add_sub_v1.tar.gz",
    )

    response = json.loads(response["Body"].read().decode("utf8"))
    output = response["outputs"][0]["data"]

    print(output)


def test():

    s3_client = boto3.client("s3")
    auto_scaling_client = boto3.client("application-autoscaling")
    sample_image_name = "shiba_inu_dog.jpg"
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    # sagemaker variables
    role ="arn:aws:iam::469651751916:role/sagemaker-execution-role"
    sm_client = boto3.client(service_name="sagemaker")
    runtime_sm_client = boto3.client("sagemaker-runtime")
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session())
    bucket = sagemaker_session.default_bucket()
    prefix = "resnet50-mme-gpu"

    # endpoint variables
    sm_model_name = f"{prefix}-mdl-{ts}"
    endpoint_config_name = f"{prefix}-epc-{ts}"
    endpoint_name = f"{prefix}-ep-{ts}"
    model_data_url = f"s3://{bucket}/{prefix}/"

    # account mapping for SageMaker MME Triton Image
    account_id_map = {
        "us-east-1": "785573368785",
        "us-east-2": "007439368137",
        "us-west-1": "710691900526",
        "us-west-2": "301217895009",
        "eu-west-1": "802834080501",
        "eu-west-2": "205493899709",
        "eu-west-3": "254080097072",
        "eu-north-1": "601324751636",
        "eu-south-1": "966458181534",
        "eu-central-1": "746233611703",
        "ap-east-1": "110948597952",
        "ap-south-1": "763008648453",
        "ap-northeast-1": "941853720454",
        "ap-northeast-2": "151534178276",
        "ap-southeast-1": "324986816169",
        "ap-southeast-2": "355873309152",
        "cn-northwest-1": "474822919863",
        "cn-north-1": "472730292857",
        "sa-east-1": "756306329178",
        "ca-central-1": "464438896020",
        "me-south-1": "836785723513",
        "af-south-1": "774647643957",
    }

    region = boto3.Session().region_name
    if region not in account_id_map.keys():
        raise ("UNSUPPORTED REGION")

    base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    mme_triton_image_uri = (
        "{account_id}.dkr.ecr.{region}.{base}/sagemaker-tritonserver:22.07-py3".format(
            account_id=account_id_map[region], region=region, base=base
        )
    )
    mme_triton_image_uri = '785573368785.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:22.07-py3'
    container = {"Image": mme_triton_image_uri, "ModelDataUrl": model_data_url, "Mode": "MultiModel"}
    create_model_response = sm_client.create_model(ModelName=sm_model_name, ExecutionRoleArn=role, PrimaryContainer=container)

    print("Model Arn: " + create_model_response["ModelArn"])

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.g5.xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": sm_model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    print("Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"])



    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])





if __name__ == "__main__":
    app = typer.Typer()
    app.command()(create_endpoint)
    app.command()(add_model)
    app.command()(add_model_2)
    app.command()(call_model)
    app.command()(call_model_2)
    app()
