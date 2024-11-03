import pytest
import pandas as pd
from search_engine.data_provider.base_data import PrepareData

@pytest.fixture
def setup_data():
    data_path = "test_data.csv"
    join_keys = [["country", "year"], ["country"], ["year"]]
    df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    return data_path, join_keys, df

def test_init_from_disk(mocker, setup_data):
    data_path, join_keys, df = setup_data
    mocker.patch('pandas.read_csv', return_value=df)
    prepare_data = PrepareData(data_path=data_path, join_keys=join_keys, from_disk=True)
    assert prepare_data.has_key
    assert prepare_data.data.shape == df.shape

def test_init_from_dataframe(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    assert prepare_data.has_key
    assert prepare_data.data.shape == df.shape

def test_get_data(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    assert prepare_data.get_data().equals(df)

def test_set_data(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    new_df = pd.DataFrame({
        "country": ["X", "Y", "Z"],
        "year": [2023, 2024, 2025],
        "value": [4, 5, 6]
    })
    prepare_data.set_data(new_df)
    assert prepare_data.get_data().equals(new_df)

def test_see_data(setup_data, capsys):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    prepare_data.see_data()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_features(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    assert (prepare_data.get_features() == df.columns).all()

def test_see_features(setup_data, capsys):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    prepare_data.see_features()
    captured = capsys.readouterr()
    assert "country" in captured.out

def test_get_join_keys(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    assert prepare_data.get_join_keys() == ["country_year", "country", "year"]

def test_cut_data_by_features(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    cut_df = prepare_data.cut_data_by_features(["country", "value"])
    assert (cut_df.columns == ["country", "value"]).all()

def test_cut_data_by_join_keys(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    cut_df = prepare_data.cut_data_by_join_keys(["country", "year"])
    assert (cut_df.columns == ["country", "year"]).all()

def test_data_cleaning(setup_data):
    _, join_keys, df = setup_data
    cleaned_df = PrepareData._data_cleaning(df, ["country", "year"])
    assert cleaned_df is not None

def test_get_num_cols(setup_data):
    _, join_keys, df = setup_data
    num_cols = PrepareData.get_num_cols(df, {"country", "year"})
    assert "value" in num_cols

def test_check_join_keys(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    assert prepare_data._check_join_keys(join_keys)

def test_construct_join_keys(setup_data):
    _, join_keys, df = setup_data
    prepare_data = PrepareData(data_path=None, join_keys=join_keys, from_disk=False, df=df)
    prepare_data._construct_join_keys()
    assert "country_year" in prepare_data.data.columns