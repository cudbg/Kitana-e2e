import pytest
import pandas as pd
from search_engine.data_provider.data_provider import PrepareBuyerSellers
from search_engine.data_provider.seller_data import PrepareSeller
from search_engine.data_provider.buyer_data import PrepareBuyer
from search_engine.data_provider.sellers import PrepareSellers

@pytest.fixture
def setup_buyer_seller():
    buyer_data = pd.DataFrame({
        "country": ["A", "B", "D"],
        "year": [2020, 2021, 2023],
        "value": [4, 5, 6]
    })
    seller_data = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    target_feature = "value"
    buyer = PrepareBuyer(data_path=None, join_keys=[["country", "year"]], features=["value"], target_feature=target_feature, from_disk=False, buyer_df=buyer_data)
    seller = PrepareSeller(data_path=None, join_keys=[["country", "year"]], features=["value"], from_disk=False, seller_df=seller_data)
    return buyer, seller

def test_init():
    prepare_buyer_sellers = PrepareBuyerSellers()
    assert prepare_buyer_sellers.buyer is None
    assert prepare_buyer_sellers.buyer_added == False
    assert isinstance(prepare_buyer_sellers.sellers, PrepareSellers)

def test_add_buyer(setup_buyer_seller):
    buyer, _ = setup_buyer_seller
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer(buyer)
    assert prepare_buyer_sellers.buyer_added == True
    assert prepare_buyer_sellers.buyer == buyer

def test_add_seller(setup_buyer_seller):
    buyer, seller = setup_buyer_seller
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer(buyer)
    prepare_buyer_sellers.add_seller("test_seller", seller)
    assert "test_seller" in prepare_buyer_sellers.sellers.get_sellers()

def test_add_buyer_by_path(mocker, setup_buyer_seller):
    buyer, _ = setup_buyer_seller
    mocker.patch('pandas.read_csv', return_value=buyer.data)
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer_by_path("test_buyer_data.csv", [["country", "year"]], ["value"], "value")
    assert prepare_buyer_sellers.buyer_added == True

def test_add_seller_by_path(mocker, setup_buyer_seller):
    buyer, seller = setup_buyer_seller
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    }))
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer(buyer)
    prepare_buyer_sellers.add_seller_by_path("test_seller_data.csv", [["country", "year"]], ["value"])
    assert "test_seller_data" in prepare_buyer_sellers.sellers.get_sellers()

def test_get_buyer(setup_buyer_seller):
    buyer, _ = setup_buyer_seller
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer(buyer)
    assert prepare_buyer_sellers.get_buyer() == buyer

def test_get_sellers(setup_buyer_seller):
    buyer, seller = setup_buyer_seller
    prepare_buyer_sellers = PrepareBuyerSellers()
    prepare_buyer_sellers.add_buyer(buyer)
    prepare_buyer_sellers.add_seller("test_seller", seller)
    sellers = prepare_buyer_sellers.get_sellers().get_sellers()
    assert "test_seller" in sellers
