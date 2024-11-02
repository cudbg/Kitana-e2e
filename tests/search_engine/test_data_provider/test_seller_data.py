import pytest
import pandas as pd
from search_engine.data_provider.seller_data import PrepareSeller

@pytest.fixture
def setup_seller_data():
    data_path = "test_seller_data.csv"
    join_keys = [["country", "year"], ["country"], ["year"]]
    features = ["value"]
    df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    return data_path, join_keys, features, df

def test_init_from_disk(mocker, setup_seller_data):
    data_path, join_keys, features, df = setup_seller_data
    mocker.patch('pandas.read_csv', return_value=df)
    prepare_seller = PrepareSeller(data_path=data_path, join_keys=join_keys, features=features, from_disk=True)
    assert prepare_seller.has_key
    assert prepare_seller.data.shape == df.shape

def test_init_from_dataframe(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    assert prepare_seller.has_key
    assert prepare_seller.data.shape == df.shape

def test_get_data(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    reordered_df = prepare_seller.get_data()[df.columns]
    assert reordered_df.equals(df)

def test_set_data(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    new_df = pd.DataFrame({
        "country": ["X", "Y", "Z"],
        "year": [2023, 2024, 2025],
        "value": [4, 5, 6]
    })
    prepare_seller.set_data(new_df)
    assert prepare_seller.get_data().equals(new_df)

def test_see_data(setup_seller_data, capsys):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    prepare_seller.see_data()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_features(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    sorted_features = sorted(prepare_seller.get_features())
    sorted_df_columns = sorted(df.columns)
    assert sorted_features == sorted_df_columns

def test_see_features(setup_seller_data, capsys):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    prepare_seller.see_features()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_join_keys(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    assert prepare_seller.get_join_keys() == ["country_year", "country", "year"]

def test_cut_data_by_features(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    cut_df = prepare_seller.cut_data_by_features(["country", "value"])
    assert (cut_df.columns == ["country", "value"]).all()

def test_cut_data_by_join_keys(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    cut_df = prepare_seller.cut_data_by_join_keys(["country", "year"])
    assert (cut_df.columns == ["country", "year"]).all()

def test_record_join_key_domains(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    join_key_domains = prepare_seller.record_join_key_domains()
    assert "country_year" in join_key_domains

def test_get_record_status(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    prepare_seller.record_join_key_domains()
    assert prepare_seller.get_record_status()

def test_set_join_key_domains(setup_seller_data):
    _, join_keys, features, df = setup_seller_data
    prepare_seller = PrepareSeller(data_path=None, join_keys=join_keys, features=features, from_disk=False, seller_df=df)
    prepare_seller.set_join_key_domains("country", {"A", "B"})
    assert "A" in prepare_seller.get_join_key_domains("country")
