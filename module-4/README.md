# Practice 

*** 


# H7: Kubeflow + AirFlow pipelines

## Reading list: 


- [Kubeflow pipelines Standalone Deployment](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Kubeflow Pipelines SDK API Reference](https://kubeflow-pipelines.readthedocs.io/en/)
- [How we Reduced our ML Training Costs by 78%!!](https://blog.gofynd.com/how-we-reduced-our-ml-training-costs-by-78-a33805cb00cf)
- [Leveraging the Pipeline Design Pattern to Modularize Recommendation Services](https://doordash.engineering/2021/07/07/pipeline-design-pattern-recommendation/)
- [Why data scientists shouldn’t need to know Kubernetes](https://huyenchip.com/2021/09/13/data-science-infrastructure.html)
- [Orchestration for Machine Learning](https://madewithml.com/courses/mlops/orchestration/)
- [KubernetesPodOperator](https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html)



## Task:

For this task, you will need both a training and an inference pipeline. The training pipeline should include at least the following steps: Load Training Data, Train Model, Save Trained Models. Additional steps may be added as desired. Similarly, the inference pipeline should include at least the following steps: Load Data for Inference, Load Trained Model, Run Inference, Save Inference Results. You may also add extra steps to this pipeline as needed.

- RP1: Write a README with instructions on deploying Kubeflow pipelines.
- PR2: Write a Kubeflow training pipeline.
- PR3: Write a Kubeflow inference pipeline.

- RP4: Write a README with instructions on how to deploy Airflow.
- PR5: Write an Airflow training pipeline.
- PR6: Write an Airflow inference pipeline.


## Criteria: 

- 6 PRs merged.



# H8: Dagster

## Reading list:

- [Orchestrating Machine Learning Pipelines with Dagster](https://dagster.io/blog/dagster-ml-pipelines)
- [ML pipelines for fine-tuning LLMs](https://dagster.io/blog/finetuning-llms)
- [Awesome pen source workflow engines](https://github.com/meirwah/awesome-workflow-engines)
- [A framework for real-life data science and ML](https://metaflow.org/)
- [New in Metaflow: Train at scale with AI/ML frameworks](https://outerbounds.com/blog/distributed-training-with-metaflow/)
- [House all your ML orchestration needs](https://flyte.org/machine-learning)


## Task:

For this task, you will need both a training and an inference pipeline. The training pipeline should include at least the following steps: Load Training Data, Train Model, Save Trained Models. Additional steps may be added as desired. Similarly, the inference pipeline should include at least the following steps: Load Data for Inference, Load Trained Model, Run Inference, Save Inference Results. You may also add extra steps to this pipeline as needed.

- Update the Google Doc with the pipeline section for your use case, and compare Kubeflow, Airflow, and Dagster.
- PR1: Write a Dagster training pipeline.
- PR2: Write a Dagster inference pipeline.

## Criteria:


- 2 PRs merged.
- Pipeline section in the google doc.

*** 

# Reference implementation

*** 



# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-4
```

Run k9s 

```
k9s -A
```

# Airflow

## Deploy airflow locally



```
export AIRFLOW_HOME=$PWD/airflow-home
```

```
AIRFLOW_VERSION=2.7.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow-providers-cncf-kubernetes==7.9.0
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

