# Module 3

![alt text](./../docs/experiments.jpg)

# Practice 

[Practice task](./PRACTICE.md)

*** 

# Reference implementation

*** 


# Project stucture 

- [Python project](https://github.com/navdeep-G/samplemod.git)
- [ML project](https://github.com/ashleve/lightning-hydra-template.git)
- [Advanced features](https://github.com/Lightning-AI/lightning)

# Configuration 

[hydra](https://hydra.cc/docs/intro/)


# Example ML model with testing

[nlp-sample](./nlp-sample)

# Experiments

https://neptune.ai/blog/best-ml-experiment-tracking-tools

## AIM 

https://github.com/aimhubio/aim


```
kubectl create -f aim/deployment-aim-web.yaml
kubectl port-forward svc/my-aim-service  8080:80 --namespace default
```


# Model card

- https://github.com/ivylee/model-cards-and-datasheets
- https://arxiv.org/abs/1810.03993


# LLMs for everything


## LoRA & Peft

- https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
- https://github.com/huggingface/peft

## Experiments 

- https://github.com/georgian-io/LLM-Finetuning-Hub
- https://medium.com/georgian-impact-blog/the-practical-guide-to-llms-llama-2-cdf21d540ce3
 
## Run example

```
python lora_training/mistral_classification.py training-llm --pretrained-ckpt mistralai/Mistral-7B-v0.1 --epochs 1 --train-sample-fraction 0.3
python lora_training/mistral_classification.py training-llm --pretrained-ckpt facebook/opt-350m --epochs 1 --train-sample-fraction 0.3

python lora_training/mistral_classification.py inference-llm
```


https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb

## Run example RLHF


```
docker build -t rlhf:latest .
docker run --net=host --gpus all -it -v ${PWD}:/main rlhf:latest /bin/bash

accelerate config
python sft_llama2.py

```

https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts


## Eval:

- https://github.com/explodinggradients/ragas
- https://github.com/NVIDIA/NeMo-Guardrails
- https://github.com/guardrail-ml/guardrail
- https://github.com/promptfoo/promptfoo
- https://github.com/confident-ai/deepeval



```
pip install nemoguardrails
pip install openai
export OPENAI_API_KEY=**********
```



# Distributed training 

- https://www.anyscale.com/blog/what-is-distributed-training
- https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray
- https://huggingface.co/docs/transformers/perf_train_gpu_many
- https://github.com/microsoft/DeepSpeed


# Hyperparameter search & AutoML

- https://github.com/microsoft/nni
- https://github.com/autogluon/autogluon


# Declarative ML

https://predibase.com/blog/how-to-fine-tune-llama-2-on-your-data-with-scalable-llm-infrastructure