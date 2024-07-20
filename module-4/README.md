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

Install

```bash
AIRFLOW_VERSION=2.9.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow-providers-cncf-kubernetes==8.3.3
```

Run standalone airflow

```bash
export AIRFLOW_HOME=./airflow_pipelines
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export WANDB_PROJECT=****************
export WANDB_API_KEY=****************
airflow standalone
```

Create storage

```bash
kubectl create -f ./airflow_pipelines/volumes.yaml
```

Open UI

```bash
open http://0.0.0.0:8080
```

Trigger training job.

```bash
airflow dags trigger training_dag
```

Trigger inference job.

```bash
airflow dags trigger inference_dag
```

### References:

- [AI + ML examples of DAGs](https://registry.astronomer.io/dags?categoryName=AI+%2B+Machine+Learning&limit=24&sorts=updatedAt%3Adesc)
- [Pass data between tasks](https://www.astronomer.io/docs/learn/airflow-passing-data-between-tasks)


# Kubeflow pipelines

Install

```bash
export WANDB_PROJECT=****************
export WANDB_API_KEY=****************
export PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```


Access UI and minio

```bash
kubectl port-forward --address=0.0.0.0 svc/minio-service 9000:9000 -n kubeflow
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow
```

Create training job.

```bash
python kfp-training-pipeline_v2.py http://0.0.0.0:8080
```

Create inference job.

```bash
python kfp-inference-pipeline_v2.py http://0.0.0.0:8080
```


### References

- https://www.kubeflow.org/docs/components/pipelines/v2/data-types/artifacts/#new-pythonic-artifact-syntax



# Dagster


- https://github.com/dagster-io/dagster_llm_finetune
- https://dagster.io/blog/finetuning-llms

