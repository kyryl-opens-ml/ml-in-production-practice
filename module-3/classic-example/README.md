# NLP sample

## Setup

```bash
make build
```

## Develop

```bash
make run_dev
cd /main
export PYTHONPATH=.
export WANDB_PROJECT=ml-in-production-practice
export WANDB_API_KEY=***********************
```

## Test

```bash
make test
```

reference: https://madewithml.com/courses/mlops/testing/

## Run training job

```bash
modal deploy run_training_job.py
python run_training_job.py
```

## Reports

```bash
open https://wandb.ai/truskovskiyk/ml-in-production-practice
```
