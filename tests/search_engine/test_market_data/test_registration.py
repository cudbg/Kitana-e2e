import pytest
from search_engine.market.data_market import DataMarket
from search_engine.config.config import get_config

def test_register_seller(data_market, sample_data):
    seller_id = data_market.register_seller(
        seller_df=sample_data["seller_df"],
        seller_name="test_seller",
        join_keys=sample_data["join_keys"],
        join_key_domains=sample_data["join_key_domains"]
    )
    
    assert seller_id == 0
    assert len(data_market.seller_sketches) > 0
    assert data_market.seller_id == 1

def test_register_buyer(data_market, sample_data):
    config = get_config()
    buyer_id = data_market.register_buyer(
        buyer_df=sample_data["buyer_df"],
        join_keys=sample_data["join_keys"],
        join_key_domains=sample_data["join_key_domains"],
        target_feature="target",
        fit_by_residual=config.search.fit_by_residual,
    )
    
    assert buyer_id == 0
    assert len(data_market.buyer_sketches) > 0
    assert data_market.buyer_id == 1