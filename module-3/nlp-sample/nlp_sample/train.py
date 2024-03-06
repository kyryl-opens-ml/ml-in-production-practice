import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from nlp_sample.config import DataTrainingArguments, ModelArguments
from nlp_sample.utils import compute_metrics, preprocess_function_examples, setup_logger

logger = logging.getLogger(__name__)


def get_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def read_dataset(data_args: DataTrainingArguments, cache_dir: str):
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    return raw_datasets, num_labels, label_list


def get_models(model_args, num_labels):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    return config, tokenizer, model


def process_dataset(data_args, label_list, model, config, tokenizer, training_args, raw_datasets):
    padding = "max_length" if data_args.pad_to_max_length else False
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    preprocess_function = partial(
        preprocess_function_examples,
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        label_to_id=label_to_id,
    )
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    return train_dataset, eval_dataset


def get_trainer(model, train_dataset, data_args, training_args, eval_dataset, tokenizer) -> Trainer:
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def get_config(config_path: Path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(config_path)
    return model_args, data_args, training_args


def train(config_path: Path):
    model_args, data_args, training_args = get_config(config_path)
    setup_logger(logger)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    raw_datasets, num_labels, label_list = read_dataset(data_args=data_args, cache_dir=model_args.cache_dir)
    config, tokenizer, model = get_models(model_args=model_args, num_labels=num_labels)
    
    train_dataset, eval_dataset = process_dataset(
        data_args=data_args,
        label_list=label_list,
        model=model,
        config=config,
        tokenizer=tokenizer,
        training_args=training_args,
        raw_datasets=raw_datasets,
    )

    trainer = get_trainer(
        model=model,
        train_dataset=train_dataset,
        data_args=data_args,
        training_args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
        "language": "en",
        "dataset_tags": "cola",
    }

    trainer.create_model_card(**kwargs)
    logger.info(f"Generate model card {training_args.output_dir}/README.md")
