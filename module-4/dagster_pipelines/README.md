---
base_model: microsoft/Phi-3-mini-4k-instruct
library_name: peft
license: mit
tags:
- generated_from_trainer
model-index:
- name: phi-3-mini-lora-text2sql
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/truskovskiyk/ml-in-production-practice/runs/7932655z)
# phi-3-mini-lora-text2sql

This model is a fine-tuned version of [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8630

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0      | 0    | 2.8867          |
| 1.0694        | 2.1436 | 500  | 0.8630          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.42.3
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.19.1