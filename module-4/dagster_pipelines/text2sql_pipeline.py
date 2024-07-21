import logging
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
import logging
import sys
from pathlib import Path

import datasets
import transformers
import wandb
from pathlib import Path
from random import randrange
import pandas as pd
from datasets import DatasetDict, load_dataset
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import modal
import evaluate
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import pipeline
from dagster import Config, asset, MetadataValue, AssetExecutionContext, asset_check, AssetCheckResult, Definitions
from random import randint

logger = logging.getLogger()



def _get_sql_data(random_state: int = 42, subsample: float = None) -> DatasetDict:
    dataset_name = "b-mc2/sql-create-context"
    dataset = load_dataset(dataset_name, split="train")
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])

    if subsample is not None:
        dataset = dataset.shuffle(seed=random_state).select(
            range(int(len(dataset) * subsample))
        )
        print(f"dataset new size: {len(dataset)}")

    datasets = dataset.train_test_split(test_size=0.05, seed=random_state)
    return datasets

@asset(group_name="data", compute_kind="python")
def load_sql_data(context: AssetExecutionContext):
    subsample = 0.1
    dataset = _get_sql_data(subsample=subsample)

    context.add_output_metadata(
        {
            "len_train": MetadataValue.int(len(dataset['train'])),
            "len_test": MetadataValue.int(len(dataset['test'])),
            "sample_train": MetadataValue.json(dataset['train'][randint(0, len(dataset['train']))]),
            "sample_test": MetadataValue.json(dataset['test'][randint(0, len(dataset['test']))]),
        }
    )

    return dataset

@asset_check(asset=load_sql_data)
def no_empty(load_sql_data):
    train_no_no_empty = len(load_sql_data['train']) != 0
    test_no_no_empty = len(load_sql_data['test']) != 0
    return AssetCheckResult(passed=train_no_no_empty and test_no_no_empty)


def create_message_column(row):
    messages = []
    user = {"content": f"{row['context']}\n Input: {row['question']}", "role": "user"}
    messages.append(user)
    assistant = {"content": f"{row['answer']}", "role": "assistant"}
    messages.append(assistant)
    return {"messages": messages}


def format_dataset_chatml(row, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=False, tokenize=False
        )
    }

@asset(group_name="data", compute_kind="python")
def process_dataset(context: AssetExecutionContext, load_sql_data) -> DatasetDict:
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    dataset = load_sql_data

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(partial(format_dataset_chatml, tokenizer=tokenizer))

    context.add_output_metadata(
        {
            "len_train": MetadataValue.int(len(dataset_chatml['train'])),
            "len_test": MetadataValue.int(len(dataset_chatml['test'])),
            "sample_train": MetadataValue.json(dataset_chatml['train'][randint(0, len(dataset_chatml['train']))]),
            "sample_test": MetadataValue.json(dataset_chatml['test'][randint(0, len(dataset_chatml['test']))]),
        }
    )

    return dataset_chatml


def get_model(model_id: str, device_map):
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    else:
        compute_dtype = torch.float16
        attn_implementation = "sdpa"

        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
        print(attn_implementation)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, add_eos_token=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    return tokenizer, model



@dataclass
class ModelArguments:
    model_id: str
    model_name: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def train_model(dataset_chatml):
    config = {

        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "model_name": "dagster-model",

        "output_dir": "/tmp/phi-3-mini-lora-text2sql",
        "eval_strategy": "steps",
        "do_eval": True,
        "optim": "adamw_torch",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "num_train_epochs": 0.01,
        "warmup_ratio": 0.1,
        "logging_first_step": True,
        "logging_steps": 500,
        "save_steps": 500,
        "seed": 42,
        "bf16": True,
        "fp16": False,
        "eval_steps": 500,
        "report_to": [
            "wandb"
        ],
        "lr_scheduler_type": "linear",
        "log_level" : "debug",
        "evaluation_strategy": "steps",
        "eval_on_start": True
        }
    setup_logger(logger)

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_dict(config)

    logger.info(f"model_args = {model_args}")
    logger.info(f"training_args = {training_args}")

    device_map = {"": 0}
    target_modules = [
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]

    set_seed(training_args.seed)
    logger.info(dataset_chatml["train"][0])

    tokenizer, model = get_model(model_id=model_args.model_id, device_map=device_map)
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml["train"],
        eval_dataset=dataset_chatml["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model()
    trainer.create_model_card()

    uri = upload_to_registry(model_name=model_args.model_name, model_path=Path(training_args.output_dir))
    return f"{model_args.model_name}:latest", uri


def setup_logger(logger):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "README.md")
        art.add_file(model_path / "adapter_config.json")
        art.add_file(model_path / "adapter_model.safetensors")
        art.add_file(model_path / "special_tokens_map.json")
        art.add_file(model_path / "tokenizer.json")
        art.add_file(model_path / "tokenizer_config.json")
        art.add_file(model_path / "training_args.bin")
        run.log_artifact(art)
        uri = run.get_url()
    return uri


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


def evaluate_model(df: pd.DataFrame, model_name: str):
    model = Predictor.from_wandb(model_name=model_name)

    generated_sql = []
    for idx in tqdm(range(len(df))):
        context = df.iloc[idx]["context"]
        question = df.iloc[idx]["question"]

        sql = model.predict(question=question, context=context)
        generated_sql.append(sql)

    gt_sql = df["answer"].values
    rouge = evaluate.load("rouge")
    metrics = rouge.compute(predictions=generated_sql, references=gt_sql)
    return metrics 



@asset(group_name="model", compute_kind="modal")
def trained_model(process_dataset):
    # local
    # model_name, uri = train_model(dataset_chatml=process_dataset)

    # modal
    process_dataset_pandas = {'train': process_dataset['train'].to_pandas(), 'test': process_dataset['test'].to_pandas()}
    model_training_job = modal.Function.lookup("ml-in-production-practice-dagster-pipeline", "training_job")
    model_name, uri = model_training_job.remote(dataset_chatml_pandas=process_dataset_pandas)

    return model_name


@asset(group_name="model", compute_kind="modal")
def model_metrics(context: AssetExecutionContext, trained_model, process_dataset):
    model_path = f"/tmp/{trained_model}"
    load_from_registry(model_name=trained_model, model_path=model_path)
    # local
    # metrics = evaluate_model(df=process_dataset['test'].to_pandas(), model_name=trained_model)

    # modal
    model_evaluate_job = modal.Function.lookup("ml-in-production-practice-dagster-pipeline", "evaluation_job")
    metrics = model_evaluate_job.remote(df=process_dataset['test'].to_pandas(), model_name=trained_model)


    context.add_output_metadata(
        {
            "results": MetadataValue.json(metrics),
        }
    )

    return metrics

@asset_check(asset=model_metrics)
def rouge1_check(model_metrics):
    return AssetCheckResult(passed=bool(model_metrics['rouge1'] > 0.8))

@asset_check(asset=model_metrics)
def rouge2_check(model_metrics):
    return AssetCheckResult(passed=bool(model_metrics['rouge2'] > 0.8))

@asset_check(asset=model_metrics)
def rougeL_check(model_metrics):
    return AssetCheckResult(passed=bool(model_metrics['rougeL'] > 0.8))

@asset_check(asset=model_metrics)
def rougeLsum_check(model_metrics):
    return AssetCheckResult(passed=bool(model_metrics['rougeLsum'] > 0.8))


defs = Definitions(assets=[load_sql_data, process_dataset, trained_model, model_metrics], asset_checks=[no_empty, rouge1_check, rouge2_check, rougeL_check, rougeLsum_check])





class Predictor:

    @classmethod
    def from_wandb(cls, model_name: str) -> 'Predictor':
        model_path = f"/tmp/{model_name}"
        load_from_registry(model_name=model_name, model_path=model_path)
        return cls(model_load_path=model_path)

    def __init__(self, model_load_path: str):
        device_map = {"": 0}
        new_model = AutoPeftModelForCausalLM.from_pretrained(
            model_load_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        merged_model = new_model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(
            model_load_path, trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
        self.pipe = pipe

    @torch.no_grad()
    def predict(self, question: str, context: str) -> str:
        pipe = self.pipe

        messages = [{"content": f"{context}\n Input: {question}", "role": "user"}]

        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            num_beams=1,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            max_time=180,
        )
        sql = outputs[0]["generated_text"][len(prompt) :].strip()
        return sql





