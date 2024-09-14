# Module 6

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
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=$WANDB_API_KEY
```


# Benchmarking

NOTE: **Premature optimization is the root of all evil!**

Deploy API from module 5

```
kubectl create -f ./k8s/app-fastapi.yaml
kubectl create -f ./k8s/app-triton.yaml
kubectl create -f ./k8s/app-streamlit.yaml
kubectl create -f ./k8s/kserve-inferenceserver.yaml
```

```
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8080:8080
```

Run load test via locust

```
locust -f load-testing/locustfile.py --host=http://0.0.0.0:8080 --users 50 --spawn-rate 10 --autostart --run-time 600s
```

Run load test via k6

```
K6_WEB_DASHBOARD=true k6 run ./load-testing/load_test.js
```

Run on k8s 

```
kubectl create -f ./k8s/vegeta-job.yaml
```

- https://github.com/locustio/locust
- https://github.com/grafana/k6
- https://github.com/gatling/gatling
- https://ghz.sh/
- https://github.com/tsenart/vegeta


# Vertical scaling

- https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler
- https://docs.railway.app/reference/scaling 

# Horizontal scaling

- https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/

Install metric server 

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Update deployment 

```
kubectl apply -f k8s/app-fastapi-resources.yaml
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
- https://kserve.github.io/website/master/modelserving/autoscaling/autoscaling/


KNative autoscaling: https://kserve.github.io/website/master/modelserving/autoscaling/autoscaling/

```
kubectl create -f ./k8s/kserve-inferenceserver-autoscaling.yaml
```


```
seq 1 1000 | xargs -n1 -P10 -I {} curl -v -H "Host: custom-model-autoscaling.default.example.com" \
-H "Content-Type: application/json" \
"http://localhost:8080/v1/models/custom-model:predict" \
-d @data-samples/kserve-input.json
```

# Async inferece 


Simple example 

```
modal deploy ./queue/simple_queue.py
python queue/simple_queue.py
```


Seldon V2 Examples: https://docs.seldon.io/projects/seldon-core/en/v2/contents/architecture/index.html
SQS: https://github.com/poundifdef/smoothmq 




## Kafka


Install kafka 

```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install zookeeper bitnami/zookeeper --set replicaCount=1 --set auth.enabled=false --set allowAnonymousLogin=true \
  --set persistance.enabled=false --version 11.0.0
helm install kafka bitnami/kafka --set zookeeper.enabled=false --set replicaCount=1 --set persistance.enabled=false \
  --set logPersistance.enabled=false --set externalZookeeper.servers=zookeeper-headless.default.svc.cluster.local \
  --version 21.0.0

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
