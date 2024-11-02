import pytest
import pandas as pd
from search_engine.data_provider.sellers import PrepareSellers
from search_engine.data_provider.seller_data import PrepareSeller
from search_engine.data_provider.buyer_data import PrepareBuyer

@pytest.fixture
def setup_sellers():
    seller_data = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    buyer_data = pd.DataFrame({
        "country": ["A", "B", "D"],
        "year": [2020, 2021, 2023],
        "value": [4, 5, 6]
    })
    target_feature = "value"
    seller = PrepareSeller(data_path=None, join_keys=[["country", "year"]], features=["value"], from_disk=False, seller_df=seller_data)
    buyer = PrepareBuyer(data_path=None, join_keys=[["country", "year"]], features=["value"], target_feature=target_feature, from_disk=False, buyer_df=buyer_data)
    return seller, buyer

def test_init():
    prepare_sellers = PrepareSellers()
    assert prepare_sellers.sellers == {}
    assert prepare_sellers.join_keys == []
    assert prepare_sellers.join_key_domains == {}

def test_add_sellers(setup_sellers):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    result = prepare_sellers.add_sellers("test_seller", seller, buyer)
    assert result == True
    assert "test_seller" in prepare_sellers.sellers

def test_add_seller_by_path(mocker, setup_sellers):
    seller, buyer = setup_sellers
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    }))
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_seller_by_path("test_seller_data.csv", [["country", "year"]], buyer, ["value"])
    assert "test_seller_data" in prepare_sellers.sellers

def test_get_sellers(setup_sellers):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_sellers("test_seller", seller, buyer)
    sellers = prepare_sellers.get_sellers()
    assert "test_seller" in sellers

def test_see_sellers(setup_sellers, capsys):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_sellers("test_seller", seller, buyer)
    prepare_sellers.see_sellers()
    captured = capsys.readouterr()
    assert "test_seller" in captured.out

def test_get_domain(setup_sellers):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_sellers("test_seller", seller, buyer)
    domain = prepare_sellers.get_domain("country_year")
    assert "A_2020" in domain

def test_update_domain(setup_sellers):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_sellers("test_seller", seller, buyer)
    prepare_sellers.update_domain(seller)
    assert "country_year" in prepare_sellers.join_key_domains

def test_get_seller_names(setup_sellers):
    seller, buyer = setup_sellers
    prepare_sellers = PrepareSellers()
    prepare_sellers.add_sellers("test_seller", seller, buyer)
    seller_names = prepare_sellers.get_seller_names()
    assert "test_seller" in seller_names
