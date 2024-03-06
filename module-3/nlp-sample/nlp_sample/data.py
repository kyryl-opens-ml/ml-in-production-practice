from pathlib import Path

from datasets import load_dataset
from datasets.arrow_reader import ArrowReader
from sklearn.model_selection import train_test_split


def load_cola_data(path_to_save: Path, random_state: int = 42):
    path_to_save.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("glue", "cola")
    df_all = ArrowReader.read_table(dataset.cache_files["train"][0]["filename"]).to_pandas()
    df_test = ArrowReader.read_table(dataset.cache_files["test"][0]["filename"]).to_pandas()

    df_train, df_val = train_test_split(df_all, random_state=random_state)

    df_train.to_csv(path_to_save / "train.csv", index=False)
    df_val.to_csv(path_to_save / "val.csv", index=False)
    df_test.to_csv(path_to_save / "test.csv", index=False)


def load_cola_data_file_input(path_to_train: Path, path_to_val: Path, path_to_test: Path, random_state: int = 42):
    path_to_train.parent.mkdir(parents=True, exist_ok=True)
    path_to_val.parent.mkdir(parents=True, exist_ok=True)
    path_to_test.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("glue", "cola")
    df_all = ArrowReader.read_table(dataset.cache_files["train"][0]["filename"]).to_pandas()
    df_test = ArrowReader.read_table(dataset.cache_files["test"][0]["filename"]).to_pandas()

    df_train, df_val = train_test_split(df_all, random_state=random_state)

    df_train.to_csv(path_to_train, index=False)
    df_val.to_csv(path_to_val, index=False)
    df_test.to_csv(path_to_test, index=False)
