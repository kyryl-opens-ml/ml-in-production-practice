# Module 6

![alt text](./../docs/serving.jpg)

# Practice 

[Practice task](./PRACTICE.md)

*** 

# Reference implementation

*** 



# Setup 

Create kind cluster 

```
export WANDB_API_KEY="cb86168a2e8db7edb905da69307450f5e7867d66"
kind create cluster --name ml-in-production-course-week-6
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=cb86168a2e8db7edb905da69307450f5e7867d66
```

Run k9s 

```
k9s -A
```


# Load test 

Deploy API 

```
kubectl create -f ./k8s/fastapi-app.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```

Run test 

```
locust -f load-testing/locustfile.py --host=http://app-fastapi.default.svc.cluster.local:8080 --users 50 --spawn-rate 10 --autostart --run-time 600s
```

Run on k8s 


```
kubectl create -f ./k8s/fastapi-locust.yaml
kubectl port-forward --address 0.0.0.0 pod/load-fastapi-naive 8089:8089
```

- https://github.com/locustio/locust
- https://github.com/grafana/k6
- https://github.com/gatling/gatling

# HPA



Install metric server 

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Create from cli

```
kubectl autoscale deployment app-fastapi --cpu-percent=50 --min=1 --max=10
```

Create from yaml

```
kubectl create -f ./k8s/fastapi-hpa.yaml
```


- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/


# Async inferece 

## Install KServe

Install kserve

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.10/hack/quick_install.sh" | bash
```

## Test single model 

```
kubectl create namespace kserve-test
kubectl create -n kserve-test -f ./k8s/kserve-iris.yaml
kubectl get inferenceservices sklearn-iris -n kserve-test
kubectl get svc istio-ingressgateway -n istio-system

kubectl port-forward --address 0.0.0.0 svc/istio-ingressgateway -n istio-system 8080:80

```

```
curl -v -H "Host: sklearn-iris.kserve-test.example.com" "http://0.0.0.0:8080/v1/models/sklearn-iris:predict" -d @data/iris-input.json
```


```
kubectl create -f load-testing/perf.yaml -n kserve-test
```


## Test custom model 


Run locally 

```
docker build -t kyrylprojector/kserve-custom:latest -f Dockerfile --target app-kserve .
docker build -t kyrylprojector/kserve-custom:latest -f Dockerfile --target app-kserve . && docker push kyrylprojector/kserve-custom:latest

docker run -e PORT=8080 -e WANDB_API_KEY=******* -p 8080:8080 kyrylprojector/kserve-custom:latest 


curl localhost:8080/v1/models/kserve-custom:predict -d @data/text-input.json
```

Run on k8s 

```
kubectl apply -f k8s/kserve-custom.yaml

kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
curl -v -H "Host: custom-model.default.example.com" "http://0.0.0.0:8080/v1/models/kserve-custom:predict" -d @data/text-input.json
```

- https://kserve.github.io/website/0.10/modelserving/v1beta1/custom/custom_model/#implement-custom-model-using-kserve-api


## Kafka


Install kafka 

```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install zookeeper bitnami/zookeeper --set replicaCount=1 --set auth.enabled=false --set allowAnonymousLogin=true --set persistance.enabled=false --version 11.0.0
helm install kafka bitnami/kafka --set zookeeper.enabled=false --set replicaCount=1 --set persistance.enabled=false --set logPersistance.enabled=false --set externalZookeeper.servers=zookeeper-headless.default.svc.cluster.local --version 21.0.0
```

Install eventing

```
kubectl apply -f https://github.com/knative/eventing/releases/download/knative-v1.9.7/eventing-crds.yaml
kubectl apply -f https://github.com/knative/eventing/releases/download/knative-v1.9.7/eventing-core.yaml
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka/releases/download/knative-v1.9.1/source.yaml
```


Install minio & creds

```
kubectl apply -f k8s/kafka-infra.yaml
```

## Kafka UI 

```
helm repo add kafka-ui https://provectus.github.io/kafka-ui
helm install helm-release-name kafka-ui/kafka-ui -f k8s/kafka-ui-values.yml

export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=kafka-ui,app.kubernetes.io/instance=helm-release-name" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward --address 0.0.0.0 $POD_NAME 8080:8080
```

## New example 

Configure minio

```
kubectl port-forward $(kubectl get pod --selector="app=minio" --output jsonpath='{.items[0].metadata.name}') 9000:9000

mc config host add myminio http://127.0.0.1:9000 minio minio123

mc mb myminio/input
mc mb myminio/output

mc admin config set myminio notify_kafka:1 tls_skip_verify="off"  queue_dir="" queue_limit="0" sasl="off" sasl_password="" sasl_username="" tls_client_auth="0" tls="off" client_tls_cert="" client_tls_key="" brokers="kafka-headless.default.svc.cluster.local:9092" topic="test" version=""


mc admin service restart myminio
mc event add myminio/input arn:minio:sqs::1:kafka -p --event put --suffix .json

```

Deploy model 

```
kubectl create -f ./k8s/kafka-model-new.yaml


# docker build -t kyrylprojector/kserve-custom-transformer:latest -f Dockerfile --target app-kserve-transformer . && docker push kyrylprojector/kserve-custom-transformer:latest
# kubectl delete -f mnist_kafka_new.yaml
# kubectl create -f mnist_kafka_new.yaml
```

Trigger the model 

```
mc cp data/text-input.json myminio/input
```


## Model optimization

- https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation
- https://github.com/huggingface/distil-whisper/


- https://github.com/intel/neural-compressor
- https://github.com/neuralmagic/sparseml


- https://github.com/huggingface/optimum-nvidia
- https://github.com/huggingface/optimum-neuron
- https://github.com/huggingface/optimum-intel
- https://github.com/huggingface/optimum-habana
- https://github.com/huggingface/optimum-amd

- https://github.com/huggingface/text-generation-inference
- https://github.com/huggingface/text-embeddings-inference

- https://github.com/Dao-AILab/flash-attention
- https://github.com/vllm-project/vllm
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/huggingface/quanto
