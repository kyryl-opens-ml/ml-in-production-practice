# Practice

Benchmark and scale your model server, then explore optimization
techniques such as quantization.

### Key tasks

- Implement dynamic batching and ensembles.
- Benchmark REST and gRPC performance.
- Apply quantization or pruning.

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
- [Dynamic Batching](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#what-is-dynamic-batching)
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
