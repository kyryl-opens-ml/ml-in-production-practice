# Practice 

*** 

# H11: Advanced features & benchmarking

## Reading list: 


- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Modal Web endpoints](https://modal.com/docs/guide/webhooks)
- [Deploy your side-projects at scale for basically nothing - Google Cloud Run](https://alexolivier.me/posts/deploy-container-stateless-cheap-google-cloud-run-serverless/)
- [K6 load tests](https://github.com/grafana/k6)
- [Locust load tests](https://github.com/locustio/locust)
- [Vegeta load tests](https://github.com/tsenart/vegeta)
- [About Simple gRPC benchmarking and load testing tool](https://github.com/bojand/ghz)
- [Most Effective Types of Performance Testing](https://loadninja.com/articles/performance-test-types/)
- [Reproducible Performance Metrics for LLM inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)
- [MLPerf Inference: Datacenter Benchmark Suite Results](https://mlcommons.org/benchmarks/inference-datacenter/)
- [MLPerf Inference Benchmark](https://arxiv.org/pdf/1911.02549.pdf)
- [ModelMesh Serving](https://kserve.github.io/website/master/modelserving/mms/modelmesh/overview/#learn-more)
- [Machine Learning Deployment: Shadow Mode](https://alexgude.com/blog/machine-learning-deployment-shadow-mode/)
- [Dynamic Batching ](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#what-is-dynamic-batching)
- [Model Ensembles](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles)
- [HTTP/REST and GRPC Protocol](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/README.html)
- [Pipelines](https://docs.seldon.io/projects/seldon-core/en/v2/contents/pipelines/index.html)
- [3 Reasons Why Data Scientists Should Care About GRPC For Serving Models](https://bentoml.com/blog/3-reasons-for-grpc)


## Task:


- PR1: Write code for dynamic request batching for your model (you might use Triton, Seldon, or KServe for this).
- PR2: Write code for an ensemble of several models (you might use Triton, Seldon, or KServe for this).
- PR3: Write code for gRPC inference for your model server (you might use Triton, Seldon, or KServe for this).
- PR4: Write code for benchmarking your model server: report latency, RPS, etc.
- PR5: Write code for benchmarking your model server via REST and gRPC.
- PR6: Write code for benchmarking your model server by components: forward pass, data processing, network latency, etc.
- Update the Google doc about model inference performance and any advanced features you would need for your model.


## Criteria: 

- 6 PRs merged 
- Model inference performance in the google doc.


# H12: Scaling infra and model

## Reading list:

- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Introduction to streaming for data scientists](https://huyenchip.com/2022/08/03/stream-processing-for-data-scientists.html)
- [SageMaker Asynchronous inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
- [Three Levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software)
- [Kubernetes Event-driven Autoscaling](https://keda.sh/)
- [REST vs Messaging for Microservices – Which One is Best?](https://solace.com/blog/experience-awesomeness-event-driven-microservices/)
- [End to end inference service example with Minio and Kafka](https://kserve.github.io/website/master/modelserving/kafka/kafka/)
- [Text Generation Inference Queue](https://github.com/huggingface/text-generation-inference/blob/main/router/src/queue.rs#L36)
- [Pending Request Count (Queue Size) Per-Model Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html)
- [Seldon and Kafka](https://docs.seldon.io/projects/seldon-core/en/v2/contents/architecture/index.html)

- [MTIA v1: Meta’s first-generation AI inference accelerator](https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/)
- [Cloud TPU v5e Inference Converter introduction](https://cloud.google.com/tpu/docs/v5e-inference-converter)
- [AWS Inferentia](https://aws.amazon.com/blogs/aws/category/artificial-intelligence/aws-inferentia/)
- [All about AI Accelerators: GPU, TPU, Dataflow, Near-Memory, Optical, Neuromorphic & more](https://www.youtube.com/watch?v=VQoyypYTz2U)


- [The Top 23 Model Compression Open Source Projects](https://awesomeopensource.com/projects/model-compression)
- [FastFormers](https://github.com/microsoft/fastformers)
- [distil-whisper](https://github.com/huggingface/distil-whisper?tab=readme-ov-file)
- [DistilBert](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://github.com/S-LoRA/S-LoRA?tab=readme-ov-file)
- [TensorFlow Model Optimization Toolkit — Pruning API](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)
- [SparseML](https://github.com/neuralmagic/sparseml)
- [Intel® Neural Compressor](https://github.com/intel/neural-compressor?tab=readme-ov-file)
- [Quantization](https://huggingface.co/docs/transformers/quantization)


## Task:

- PR1: Write code for using horizontal pod autoscaling for your pod with model server.
- PR2: Write code for async inference for your models with the help of queue (Kafka, or any other queue).
- PR3: Write code for optimize inference speed for your model with the help of pruning, distillation, and quatization.
- PR4: Write code for benchmarking your model after all optimization.
- Update the Google doc about model inference performance and any advanced features you would need for your model.

## Criteria:


- 4 PRs merged. 
- Model inference performance optimization in the google doc.


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
