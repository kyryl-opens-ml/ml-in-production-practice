from pathlib import Path

from datasets import load_dataset
from datasets.arrow_reader import ArrowReader
from sklearn.model_selection import train_test_split


def _get_cola_data(random_state: int = 42):

    dataset_name = ""
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])


    print(dataset[randrange(len(dataset))])

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)

    print(dataset_chatml[0])


    dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

    subsample = True
    if subsample:
        dataset_chatml['train'] = dataset_chatml['train'].shuffle(seed=1234).select(range(7_000))
        dataset_chatml['test'] = dataset_chatml['test'].shuffle(seed=1234).select(range(700))

    dataset = load_dataset("glue", "sst2")
    df_all = ArrowReader.read_table(
        dataset.cache_files["train"][0]["filename"]
    ).to_pandas()
    df_test = ArrowReader.read_table(
        dataset.cache_files["test"][0]["filename"]
    ).to_pandas()

    df_train, df_val = train_test_split(df_all, random_state=random_state)

    return df_train, df_val, df_test


def load_cola_data(path_to_save: Path):
    path_to_save.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = _get_cola_data()

    df_train.to_csv(path_to_save / "train.csv", index=False)
    df_val.to_csv(path_to_save / "val.csv", index=False)
    df_test.to_csv(path_to_save / "test.csv", index=False)


def load_cola_data_file_input(
    path_to_train: Path, path_to_val: Path, path_to_test: Path
):
    path_to_train.parent.mkdir(parents=True, exist_ok=True)
    path_to_val.parent.mkdir(parents=True, exist_ok=True)
    path_to_test.parent.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = _get_cola_data()

    df_train.to_csv(path_to_train, index=False)
    df_val.to_csv(path_to_val, index=False)
    df_test.to_csv(path_to_test, index=False)
