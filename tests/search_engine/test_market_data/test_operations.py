import pytest
from search_engine.sketches.sketch_buyer import BuyerSketch
from search_engine.sketches.sketch_seller import SellerSketch

def test_get_seller_sketch(data_market, sample_data):
    seller_id = data_market.register_seller(
        seller_df=sample_data["seller_df"],
        seller_name="test_seller",
        join_keys=sample_data["join_keys"],
        join_key_domains=sample_data["join_key_domains"]
    )
    
    for join_key in sample_data["join_keys"]:
        sketch = data_market.get_seller_sketch_by_keys(join_key, seller_id)
        assert isinstance(sketch, SellerSketch)

def test_get_buyer_sketch(data_market, sample_data):
    buyer_id = data_market.register_buyer(
        buyer_df=sample_data["buyer_df"],
        join_keys=sample_data["join_keys"],
        join_key_domains=sample_data["join_key_domains"],
        target_feature="target"
    )
    
    for join_key in sample_data["join_keys"]:
        sketch = data_market.get_buyer_sketch_by_keys(join_key)
        assert isinstance(sketch, BuyerSketch)

def test_reset_operations(data_market, sample_data):
    # Register buyer first
    data_market.register_buyer(
        buyer_df=sample_data["buyer_df"],
        join_keys=sample_data["join_keys"],
        join_key_domains=sample_data["join_key_domains"],
        target_feature="target"
    )
    
    # Test reset operations
    data_market.reset_buyer_sketches()
    assert len(data_market.buyer_sketches) == 0
    
    data_market.reset_buyer_id_to_df_and_name()
    assert len(data_market.buyer_id_to_df_and_name) == 0