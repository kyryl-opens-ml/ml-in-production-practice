from pathlib import Path
from typing import Tuple

import great_expectations as ge
import pandas as pd
import pytest
from classic_example.data import load_sst2_data
from great_expectations.dataset.pandas_dataset import PandasDataset


@pytest.fixture(scope="session")
def data_path() -> Path:
    _data_path = Path("/tmp/data")
    _data_path.mkdir(exist_ok=True, parents=True)

    load_sst2_data(path_to_save=_data_path)

    return _data_path


@pytest.fixture(scope="session")
def data(data_path: Path) -> Tuple[PandasDataset, PandasDataset]:
    df_train = pd.read_csv(data_path / "train.csv")
    df_val = pd.read_csv(data_path / "val.csv")
    df_test = pd.read_csv(data_path / "test.csv")

    return (
        ge.dataset.PandasDataset(df_train),
        ge.dataset.PandasDataset(df_val),
        ge.dataset.PandasDataset(df_test),
    )
