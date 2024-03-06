from pathlib import Path

import pytest
from transformers import Trainer, TrainingArguments

from nlp_sample.config import DataTrainingArguments, ModelArguments
from nlp_sample.train import get_models, get_trainer, process_dataset, read_dataset, train


@pytest.fixture()
def model_args() -> ModelArguments:
    return ModelArguments(model_name_or_path="prajjwal1/bert-tiny")


@pytest.fixture()
def data_args(data_path: Path) -> DataTrainingArguments:
    return DataTrainingArguments(
        train_file=str(data_path / "train.csv"),
        validation_file=str(data_path / "val.csv"),
        max_train_samples=4,
        max_eval_samples=2,
    )


@pytest.fixture()
def training_args() -> TrainingArguments:
    return TrainingArguments(output_dir="/tmp/test", num_train_epochs=1000, report_to=[], learning_rate=5e-04)


@pytest.fixture()
def trainer_with_one_batch(
    model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments
) -> Trainer:
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
    return trainer


@pytest.fixture()
def config_path() -> Path:
    return "tests/data/test_config.json"


def test_minimum_functionality():
    # "Bad"
    # "Good"
    pass


def test_invariance():
    # "I am fliong to NYC" -> N
    # "I am fliong to Toronto" -> N
    pass


def test_directional():
    # "Bad" - > "GOOD"
    pass


def test_overfit_batch(trainer_with_one_batch: Trainer):
    train_result = trainer_with_one_batch.train()
    metrics = train_result.metrics
    assert metrics["train_loss"] < 0.01


def test_train_to_completion(config_path: Path):
    train(config_path=config_path)
    result_path = Path("/tmp/results")
    assert result_path.exists()
    assert (result_path / "model.safetensors").exists()
    assert (result_path / "training_args.bin").exists()
    assert (result_path / "all_results.json").exists()
    assert (result_path / "README.md").exists()
