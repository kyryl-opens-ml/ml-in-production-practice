# Module 2

![alt text](./../docs/data.jpg)

# Practice

[Practice task](./PRACTICE.md)

***

# Reference implementation

***

# Minio

## Install libraries

```bash
pip install -r requirements.txt
```

## Docker setup

```bash
docker run -it -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"
```

## Kubernetes setup

Create kind cluster

```bash
kind create cluster --name ml-in-production
```

Run k9s

```bash
k9s -A
```

Deploy

```bash
kubectl create -f minio_storage/minio-standalone-dev.yaml
```

Access UI and API

```bash
kubectl port-forward --address=0.0.0.0 pod/minio 9000:9000
kubectl port-forward --address=0.0.0.0 pod/minio 9001:9001
```

see [this issue](https://github.com/minio/console/issues/2539) about UI access.

## S3 access to Minio

You can use Minio via AWS CLI

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://127.0.0.1:9000
```

AWS CLI

```bash
aws s3 ls
aws s3api create-bucket --bucket test
aws s3 cp --recursive . s3://test/
```

## Minio Client

Run unit tests with minio client.

```bash
pytest -ss ./minio_storage/test_minio_client.py
```

# Pandas profiling

<https://aaltoscicomp.github.io/python-for-scicomp/data-formats/>

![alt text](./images/pandas-formats.png)


# CVS inference performance

Run experiments.

```bash
python processing/inference_example.py run-single-worker --inference-size 10000000
python processing/inference_example.py run-pool --inference-size 10000000
python processing/inference_example.py run-ray --inference-size 10000000
```

Results.

| Name of Inference    | Time (seconds)      |
|----------------------|---------------------|
| Inference 1 worker   | 12.90  |
| Inference 16 workers | 3.88  |
| Inference with Ray   | 3.16  |


# Streaming dataset


Create

```bash
python streaming-dataset/mock_data.py create-data --path-to-save random-data
```

Upload

```bash
aws s3api create-bucket --bucket datasets
aws s3 cp --recursive random-data s3://datasets/random-data
```

Read

```bash
python streaming-dataset/mock_data.py get-dataloader --remote random-data
```

Alternatives:

- <https://www.tensorflow.org/tutorials/load_data/tfrecord>
- <https://github.com/aws/amazon-s3-plugin-for-pytorch>
- <https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/>
- <https://github.com/webdataset/webdataset>
- <https://github.com/mosaicml/streaming>
- <https://github.com/huggingface/datatrove>

# Vector Databases

Create database.

```bash
python vector-db/rag_cli_application.py create-new-vector-db --table-name test --number-of-documents 300
```

Query database.

```bash
python vector-db/rag_cli_application.py query-existing-vector-db  --query 'complex query' --table-name test
```

Storage [diagram](https://lancedb.github.io/lancedb/concepts/storage/)


# DVC

Init DVC

```
dvc init --subdir
git status
git commit -m "Initialize DVC"
```

Add data

```
mkdir data
touch ./data/big-data.csv
```

Add to dvc

```
dvc add ./data/big-data.csv
git add data/.gitignore data/big-data.csv.dvc
git commit -m "Add raw data"
```

Add remote

```
aws s3api create-bucket --bucket ml-data --endpoint-url $AWS_ENDPOINT

dvc remote add -d minio s3://ml-data
dvc remote modify minio endpointurl $AWS_ENDPOINT
```

Save code to git

```
git add .dvc/config
git commit -m "Configure remote storage"
git push 
```

Save data to storage

```
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
dvc push
```

- <https://dvc.org/doc/start/data-management>
- <https://github.com/iterative/dataset-registry>

# Labeling: Label-studio and RLHF

```
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest
```

```
python generate_mock_rlhf_dataset.py
```

Reference:

- <https://labelstud.io/blog/create-a-high-quality-rlhf-dataset/>
- <https://github.com/HumanSignal/RLHF>
- <https://github.com/huggingface/trl>
- <https://huggingface.co/google/flan-t5-base>
- <https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/feedback/train-reward-model-rlhf.html>

## Labeling: Argilla

```bash
docker run -it --rm --name argilla -p 6900:6900 argilla/argilla-quickstart:v2.0.0rc1
```

User/Password you can find [here])https://github.com/argilla-io/argilla/blob/v2.0.0rc1/argilla-server/docker/quickstart/Dockerfile#L60-L62).
