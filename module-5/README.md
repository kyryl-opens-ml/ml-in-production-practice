# Module 5

![alt text](./../docs/serving.jpg)

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


# Setup 


```
export WANDB_API_KEY='your key here'
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
curl -X POST -H "Content-Type: application/json" -d @data-samples/samples.json http://0.0.0.0:8080/predict
```

```
pytest -ss ./tests
```

# Triton Inference Server 

```
make run_pytriton
```


# LLMs


- https://github.com/vllm-project/vllm
- https://github.com/huggingface/text-generation-inference
- https://github.com/predibase/lorax
- https://github.com/triton-inference-server/vllm_backend
- https://github.com/ray-project/ray-llm


# KServe 

Install KServe

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.13/hack/quick_install.sh" | bash
```

Deploy custom model

```
kubectl create -f ./k8s/kserve-inferenceserver.yaml
```

Call API 

```
kubectl get inferenceservices custom-model
kubectl get svc --namespace istio-system
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80

curl -v -H "Host: custom-model.default.example.com" -H "Content-Type: application/json" "http://localhost:8080/v1/models/custom-model:predict" -d @data-samples/iris-input.json

```


# Seldon V2 

```
git clone https://github.com/SeldonIO/seldon-core --branch=v2
```



## IRIS

```
kubectl create -f k8s/kserve-iris.yaml
kubectl get inferenceservices sklearn-iris
```

Port forward

```
kubectl get svc --namespace istio-system
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

Call API

```
curl -v -H "Host: sklearn-iris.default.example.com" -H "Content-Type: application/json" "http://localhost:8080/v1/models/sklearn-iris:predict" -d @data-samples/iris-input.json
```

## Custom


- https://kserve.github.io/website/latest/modelserving/v1beta1/custom/custom_model/#build-custom-serving-image-with-buildpacks

```
docker build -f Dockerfile -t kyrylprojector/custom-model:latest --target app-kserve .
docker push kyrylprojector/custom-model:latest

docker run -e PORT=8080 -p 5000:8080 kyrylprojector/custom-model:latest
curl localhost:5000/v1/models/custom-model:predict -d @data-samples/kserve-input.json


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