from typing import Tuple

from great_expectations.dataset.pandas_dataset import PandasDataset


def test_data_shape(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data
    assert df_train.shape[0] + df_val.shape[0] == 8551
    assert df_train.shape[1] == df_val.shape[1] == 3
    assert df_test.shape == (1063, 3)


def test_data_order(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data
    assert df_train.expect_table_columns_to_match_ordered_list(column_list=["sentence", "label", "idx"])["success"]
    assert df_val.expect_table_columns_to_match_ordered_list(column_list=["sentence", "label", "idx"])["success"]
    assert df_test.expect_table_columns_to_match_ordered_list(column_list=["sentence", "label", "idx"])["success"]


def test_data_content(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data

    assert df_train.expect_column_values_to_not_be_null(column="sentence")["success"]
    assert df_train.expect_column_values_to_not_be_null(column="label")["success"]

    assert df_val.expect_column_values_to_not_be_null(column="sentence")["success"]
    assert df_val.expect_column_values_to_not_be_null(column="label")["success"]

    assert df_test.expect_column_values_to_not_be_null(column="sentence")["success"]
    assert df_test.expect_column_values_to_not_be_null(column="label")["success"]

    assert df_train.expect_column_values_to_be_unique(column="idx")["success"]
    assert df_test.expect_column_values_to_be_unique(column="idx")["success"]

    assert df_val.expect_column_values_to_be_unique(column="idx")["success"]
    assert df_val.expect_column_values_to_be_unique(column="idx")["success"]

    assert df_train.expect_column_values_to_be_in_set("label", [1, 0])["success"]
    assert df_test.expect_column_values_to_be_in_set("label", [-1])["success"]
