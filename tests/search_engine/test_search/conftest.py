import pytest
import pandas as pd
import torch
from search_engine.market.data_market import DataMarket
from search_engine.config.config import get_config

@pytest.fixture
def sample_market_data():
    # Prepare seller data
    seller1_df = pd.DataFrame({
        "country": ["US", "UK", "CN"],
        "year": [2020, 2021, 2022],
        "value1": [100, 200, 300],
        "value2": [10, 20, 30]
    })
    
    seller2_df = pd.DataFrame({
        "country": ["US", "UK", "CN"],
        "year": [2020, 2021, 2022],
        "price1": [50, 60, 70],
        "price2": [5, 6, 7]
    })
    
    # Prepare buyer data
    buyer_df = pd.DataFrame({
        "country": ["US", "UK"],
        "year": [2020, 2021],
        "target": [1000, 2000]
    })
    
    join_keys = ["country", "year"]
    join_key_domains = {
        "country": ["US", "UK", "CN"],
        "year": [2020, 2021, 2022]
    }
    
    return {
        "seller1_df": seller1_df,
        "seller2_df": seller2_df,
        "buyer_df": buyer_df,
        "join_keys": join_keys,
        "join_key_domains": join_key_domains
    }

@pytest.fixture
def setup_market(sample_market_data):
    config = get_config()
    market = DataMarket()
    
    # Register sellers
    market.register_seller(
        seller_df=sample_market_data["seller1_df"],
        seller_name="seller1",
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"]
    )
    
    market.register_seller(
        seller_df=sample_market_data["seller2_df"],
        seller_name="seller2", 
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"]
    )
    
    # Register buyer
    market.register_buyer(
        buyer_df=sample_market_data["buyer_df"],
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"],
        target_feature="target",
        fit_by_residual=config.search.fit_by_residual
    )
    
    return market

@pytest.fixture
def setup_market_with_residual(sample_market_data):
    config = get_config()
    market = DataMarket()
    
    # Register sellers
    market.register_seller(
        seller_df=sample_market_data["seller1_df"],
        seller_name="seller1",
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"]
    )
    
    market.register_seller(
        seller_df=sample_market_data["seller2_df"],
        seller_name="seller2", 
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"]
    )
    
    # Register buyer
    market.register_buyer(
        buyer_df=sample_market_data["buyer_df"],
        join_keys=sample_market_data["join_keys"],
        join_key_domains=sample_market_data["join_key_domains"],
        target_feature="target",
        fit_by_residual=True
    )
    
    return market