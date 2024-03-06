# Practice 

*** 


# H9: API serving

## Reading list: 

- [CS 329S Lecture 8. Model Deployment](https://docs.google.com/document/d/1hNuW6bqWYZjlwpit_8W1cu7kllb-jTfy3Liof1GJWug/edit#heading=h.kp1fg79091xd)
- [Machine Learning Systems Design](https://docs.google.com/presentation/d/1U_zKs19VLJKnGE02JDRnzxJ8lgeVF22WSZ_GrA646fY/edit#slide=id.p)
- [APIs for Model Serving](https://madewithml.com/courses/mlops/api/)
- [RESTful API Design Tips from Experience](https://github.com/peterboyer/restful-api-design-tips)
- [Developing Restful APIs: A Comprehensive Set of Guidelines by Zalando](https://github.com/zalando/restful-api-guidelines)
- [Create an app Streamlit](https://docs.streamlit.io/get-started/tutorials/create-an-app)
- [Gradio Quickstart](https://www.gradio.app/guides/quickstart) 
- [Top 6 Kubernetes Deployment Strategies and How to Choose](https://codefresh.io/learn/kubernetes-deployment/top-6-kubernetes-deployment-strategies-and-how-to-choose/)


## Task:

- PR1: Write a Streamlit UI for serving your model, with tests and CI integration.
- PR2: Write a Gradio UI for serving your model, with tests and CI integration.
- PR3: Write a FastAPI server for your model, with tests and CI integration.
- PR4: Write a Kubernetes deployment YAML (Deployment, Service) for your model's API.
- PR5: Write a Kubernetes deployment YAML (Deployment, Service) for your model's UI (Streamlit, Gradio).
- Google doc update with a model serving plan for your ML model. 

## Criteria: 

- 5 PRs merged 
- Serving plan in the google doc.


# H10: Inference servers

## Reading list:

- [Machine learning system design pattern](https://github.com/mercari/ml-system-design-pattern)
- [Seldon Core v2](https://docs.seldon.io/projects/seldon-core/en/v2/contents/about/index.html)
- [TorchServe](https://pytorch.org/serve/index.html)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit)
- [Overview of getting predictions on Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/overview)
- [Qwak Model Serving](https://www.qwak.com/platform/model-serving)
- [ModalLab Fast inference with vLLM (Mistral 7B)](https://modal.com/docs/examples/vllm_inference)
- [Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#run-falcon)
- [Easy, fast, and cheap LLM serving for everyone](https://github.com/vllm-project/vllm?tab=readme-ov-file)
- [Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs](https://github.com/predibase/lorax?tab=readme-ov-file)
- [Book: Machine Learning Systems with TinyML](https://harvard-edge.github.io/cs249r_book/)

## Task:


- PR1: Write code for Seldon API deployment of your model, including tests.
- PR2: Write code for KServe API integration with your model, including tests.
- PR3: Write code for Triton Inference Server deployment, incorporating tests.
- PR4: Write code for Ray deployment, complete with tests.
- PR5: Write code for LLM deployment using TGI, vLLM, and LoRAX.
- PR6: Write code for LLM deployment with ModalLab.
- Update the Google document on model serving, outlining options and comparisons between custom servers and inference servers. Decide and explain which solution you will use and why.


## Criteria:

- 6 PRs merged 
- Serving comparisons and conclusion in the google doc.

*** 

# Reference implementation

*** 



# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-5
```

Run k9s 

```
k9s -A
```


# Setup 


```
export WANDB_API_KEY=cb86168a2e8db7edb905da69307450f5e7867d66
```


```
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=$WANDB_API_KEY
```

# Streamlit 

Run locally: 

```
make run_app_streamlit
```


Deploy k8s: 

```
kubectl create -f k8s/app-streamlit.yaml
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8081:8080
```

# Fast API

Run locally: 

```
make run_fast_api
```

Deploy k8s: 

```
kubectl create -f k8s/app-fastapi.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8081:8080
```



# Test 

```
http POST http://0.0.0.0:8080/predict < samples.json
```

```
pytest -ss ./tests
```

# Triton 


```
docker run -v $PWD:/dev_data --shm-size=1g --ulimit memlock=-1 --net=host --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3 /bin/bash

pip install -r /dev_data/requirements.txt
export WANDB_API_KEY=cb86168a2e8db7edb905da69307450f5e7867d66

tritonserver --http-port 5000 --model-repository /dev_data/triton-python-example/

```


- https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/triton/README.md
- https://github.com/triton-inference-server/fastertransformer_backend
- https://github.com/triton-inference-server/fastertransformer_backend

# LLMs


- https://github.com/vllm-project/vllm
- https://github.com/huggingface/text-generation-inference
- https://github.com/predibase/lorax
- https://github.com/triton-inference-server/vllm_backend
- https://github.com/ray-project/ray-llm


# KServe 

Install 

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

Deploy iris

```
kubectl create -f k8s/kserve-iris.yaml
kubectl get inferenceservices sklearn-iris
```

Port forward iris

```
kubectl get svc --namespace istio-system
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

Call API

```
kubectl get inferenceservice sklearn-iris
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}' | cut -d "/" -f 3)

export SERVICE_HOSTNAME=sklearn-iris.default.example.com
export INGRESS_HOST=localhost
export INGRESS_PORT=8080

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
```

Load test 


```
kubectl create -f https://raw.githubusercontent.com/kserve/kserve/release-0.11/docs/samples/v1beta1/sklearn/v1/perf.yaml
```



Custom model 

- https://kserve.github.io/website/latest/modelserving/v1beta1/custom/custom_model/#build-custom-serving-image-with-buildpacks

```
docker build -f Dockerfile -t kyrylprojector/custom-model:latest --target app-kserve .
docker push kyrylprojector/custom-model:latest

docker run -e PORT=8080 -p 5000:8080 kyrylprojector/custom-model:latest
curl localhost:5000/v1/models/custom-model:predict -d @./kserve-input.json


kubectl create -f k8s/kserve-custom.yaml
kubectl get inferenceservice custom-model
SERVICE_HOSTNAME=$(kubectl get inferenceservice custom-model -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
curl -v -H "Host: custom-model.default.example.com" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/custom-model:predict" -d @./kserve-input.json
```



# Seldon V1


## Install with helm

```
kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml
kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

kubectl create namespace seldon-system

helm install seldon-core seldon-core-operator --version 1.15.1 --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set ambassador.enabled=true  --namespace seldon-system
```

## Port forward 

```
kubectl port-forward  --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
```

## Simple example
```
kubectl create -f k8s/seldon-iris.yaml

open http://IP:7777/seldon/default/iris-model/api/v1.0/doc/#/
{ "data": { "ndarray": [[1,2,3,4]] } }

curl -X POST "http://IP:7777/seldon/default/iris-model/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"ndarray\":[[1,2,3,4]]}}"
```

## Custom example
```
kubectl create -f k8s/seldon-custom.yaml

open http://IP:7777/seldon/default/nlp-sample/api/v1.0/doc/#/
{ "data": { "ndarray": ["this is an example"] } }


curl -X POST "http://IP:7777/seldon/default/nlp-sample/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"ndarray\":[\"this is an example\"]}}"

```