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
Kafka: https://kserve.github.io/website/master/modelserving/kafka/kafka/


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
