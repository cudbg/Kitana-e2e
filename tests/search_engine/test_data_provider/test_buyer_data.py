import pytest
import pandas as pd
from search_engine.data_provider.buyer_data import PrepareBuyer

@pytest.fixture
def setup_buyer_data():
    data_path = "test_buyer_data.csv"
    join_keys = [["country", "year"], ["country"], ["year"]]
    target_feature = "value"
    df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    return data_path, join_keys, target_feature, df

def test_init_from_disk(mocker, setup_buyer_data):
    data_path, join_keys, target_feature, df = setup_buyer_data
    mocker.patch('pandas.read_csv', return_value=df)
    prepare_buyer = PrepareBuyer(data_path=data_path, join_keys=join_keys, target_feature=target_feature, from_disk=True)
    assert prepare_buyer.has_key
    assert prepare_buyer.data.shape == df.shape

def test_init_from_dataframe(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    assert prepare_buyer.has_key
    assert prepare_buyer.data.shape == df.shape

def test_get_data(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    # Reorder columns to match the original DataFrame
    reordered_df = prepare_buyer.get_data()[df.columns]
    assert reordered_df.equals(df)

def test_set_data(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    new_df = pd.DataFrame({
        "country": ["X", "Y", "Z"],
        "year": [2023, 2024, 2025],
        "value": [4, 5, 6]
    })
    prepare_buyer.set_data(new_df)
    assert prepare_buyer.get_data().equals(new_df)

def test_see_data(setup_buyer_data, capsys):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    prepare_buyer.see_data()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_features(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    # Sort columns to match the original DataFrame
    sorted_features = sorted(prepare_buyer.get_features())
    sorted_df_columns = sorted(df.columns)
    assert sorted_features == sorted_df_columns

def test_see_features(setup_buyer_data, capsys):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    prepare_buyer.see_features()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_join_keys(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    assert prepare_buyer.get_join_keys() == ["country_year", "country", "year"]

def test_cut_data_by_features(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    cut_df = prepare_buyer.cut_data_by_features(["country", "value"])
    assert (cut_df.columns == ["country", "value"]).all()

def test_cut_data_by_join_keys(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    cut_df = prepare_buyer.cut_data_by_join_keys(["country", "year"])
    assert (cut_df.columns == ["country", "year"]).all()

def test_data_cleaning(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    cleaned_df = PrepareBuyer._data_cleaning(df, ["country", "year"])
    assert cleaned_df is not None

def test_get_num_cols(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    num_cols = PrepareBuyer.get_num_cols(df, {"country", "year"})
    assert "value" in num_cols

def test_check_join_keys(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    assert prepare_buyer._check_join_keys(join_keys)

def test_construct_join_keys(setup_buyer_data):
    _, join_keys, target_feature, df = setup_buyer_data
    prepare_buyer = PrepareBuyer(data_path=None, join_keys=join_keys, target_feature=target_feature, from_disk=False, buyer_df=df)
    prepare_buyer._construct_join_keys()
    assert "country_year" in prepare_buyer.data.columns
