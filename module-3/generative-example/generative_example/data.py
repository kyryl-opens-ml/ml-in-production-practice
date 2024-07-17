from pathlib import Path
from random import randrange

from datasets import DatasetDict, load_dataset


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


def load_sql_data(path_to_save: Path, subsample: float = None):
    path_to_save.mkdir(parents=True, exist_ok=True)

    datasets = _get_sql_data(subsample=subsample)

    datasets["train"].to_json(path_to_save / "train.json")
    datasets["test"].to_json(path_to_save / "test.json")


def load_sql_data_file_input(
    path_to_train: Path, path_to_test: Path, subsample: float = None
):
    path_to_train.parent.mkdir(parents=True, exist_ok=True)
    path_to_test.parent.mkdir(parents=True, exist_ok=True)

    datasets = _get_sql_data(subsample=subsample)

    datasets["train"].to_json(path_to_train)
    datasets["test"].to_json(path_to_test)
