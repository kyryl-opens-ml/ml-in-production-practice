# Module 5

![Model serving](./../docs/serving.jpg)

## Overview

This module walks through deploying APIs and UIs for your models and
integrating inference servers like Triton and vLLM.

## Practice

[Practice task](./PRACTICE.md)

---

## Reference implementation

---


# Setup

Create kind cluster

```bash
kind create cluster --name ml-in-production
```

Run k9s

```bash
k9s -A
```


# Setup 


```
export WANDB_API_KEY='your key here'
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
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8080:8080
```

# Fast API

Run locally: 

```
make run_fast_api
```

Deploy k8s: 

```
kubectl create -f k8s/app-fastapi.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```


# Test 

```
curl -X POST -H "Content-Type: application/json" -d @data-samples/samples.json http://0.0.0.0:8080/predict
```

```
pytest -ss ./tests
```

# Triton Inference Server 

```
make run_pytriton
```


# KServe 

Install KServe

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.13/hack/quick_install.sh" | bash
```

Deploy custom model

```
kubectl create -f ./k8s/kserve-inferenceserver.yaml
```

Port forward via istio

```
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

Call API 

```
curl -v -H "Host: custom-model.default.example.com" -H "Content-Type: application/json" "http://localhost:8080/v1/models/custom-model:predict" -d @data-samples/kserve-input.json
```


# Serving LLMs via vLLM


Run server 

```
mkdir -p vllm-storage
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve microsoft/Phi-3-mini-4k-instruct --dtype auto --max-model-len 512 --enable-lora --gpu-memory-utilization 0.8 --download-dir ./vllm-storage
```


Run client 

Get list of models:

```
python serving-llm/client.py list-of-models
```


Add custom adapter:

```
python serving-llm/client.py load-from-registry truskovskiyk/ml-in-production-practice/modal_generative_example:latest sql-default-model
python serving-llm/client.py load-adapter sql-default-model ./sql-default-model
python serving-llm/client.py list-of-models
```


Test client:

```
python serving-llm/client.py test-client microsoft/Phi-3-mini-4k-instruct
python serving-llm/client.py test-client sql-default-model
```


Deploy 

Run K8S with GPUs

```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
minikube start --driver docker --container-runtime docker --gpus all
```

Create deployment 

```
kubectl create -f ./k8s/vllm-inference.yaml
kubectl port-forward --address 0.0.0.0 svc/app-vllm 8000:8000
kubectl logs <POD> -c model-loader
kubectl logs <POD> -c app-vllm
```


## Updated design doc

[Google doc](https://docs.google.com/document/d/1ZCnnsnHHiDkc3FgK2XBVur9W7nkDA7SKoPd1pGa-irQ/edit?usp=sharing)
