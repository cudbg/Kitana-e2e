import pytest
import pandas as pd
from search_engine.market.data_market import DataMarket
from search_engine.config.config import get_config

@pytest.fixture
def sample_data():
    # Sample seller data
    seller_df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3],
        "price": [100, 200, 300]
    })
    
    # Sample buyer data
    buyer_df = pd.DataFrame({
        "country": ["A", "B"],
        "year": [2020, 2021],
        "target": [10, 20]
    })
    
    # Join key domains
    join_key_domains = {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    
    return {
        "seller_df": seller_df,
        "buyer_df": buyer_df,
        "join_key_domains": join_key_domains,
        "join_keys": ["country", "year"]
    }

@pytest.fixture
def data_market():
    config = get_config("test_config.yaml")
    return DataMarket()

@pytest.fixture
def data_market_with_residual():
    config = get_config("test_config.yaml")
    return DataMarket(device=config.search.device, fit_by_residual=True)