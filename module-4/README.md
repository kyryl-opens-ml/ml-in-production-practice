# Module 4

![alt text](./../docs/pipelines.jpg)

# Practice 

[Practice task](./PRACTICE.md)

*** 

# Reference implementation

*** 

# Setup 

Create kind cluster 

```bash
kind create cluster --name ml-in-production
```

Run k9s

```bash
k9s -A
```

# Airflow

## Deploy airflow locally

```bash
export AIRFLOW_HOME=$PWD/airflow_pipelines
```

```bash
AIRFLOW_VERSION=2.9.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow-providers-cncf-kubernetes==8.3.3
```

1. Run standalone airflow

```
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

2. Create storage

```
kubectl create -f airflow-volumes.yaml
```

3. Read to run pipelines

- https://madewithml.com/courses/mlops/orchestration/


### References:

- https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html
- https://www.astronomer.io/guides/kubepod-operator/
- https://www.astronomer.io/guides/airflow-passing-data-between-tasks/


# Kubeflow pipelines 

## Deploy kubeflow pipelines 

Create directly

```
export PIPELINE_VERSION=2.0.3
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

Create yaml and applay with kubectl (better option)

```
export PIPELINE_VERSION=2.0.3
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" > kfp-yml/res.yaml
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION" > kfp-yml/pipelines.yaml

kubectl create -f kfp-yml/res.yaml
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl create -f kfp-yml/pipelines.yaml
```


Access UI and minio


```
kubectl port-forward --address=0.0.0.0 svc/minio-service 9000:9000 -n kubeflow
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow
```


## Create pipelines

Setup env variables 

```
export WANDB_PROJECT=****************
export WANDB_API_KEY=****************
```


### Training & Inference V2 (2.0.3)

```
python kfp-training-pipeline_v2.py http://0.0.0.0:8080
```

```
python kfp-inference-pipeline_v2.py http://0.0.0.0:8080
```


### Training & Inference V1 (1.8.9)


```
python kfp-training-pipeline_v1.py http://0.0.0.0:8080
```

```
python kfp-inference-pipeline_v1.py http://0.0.0.0:8080
```

### References

- https://www.kubeflow.org/docs/components/pipelines/v2/data-types/artifacts/#new-pythonic-artifact-syntax



# Dagster


- https://github.com/dagster-io/dagster_llm_finetune
- https://dagster.io/blog/finetuning-llms

