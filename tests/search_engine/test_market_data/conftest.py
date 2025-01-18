import pytest
import pandas as pd
from search_engine.market.data_market import DataMarket

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
    return DataMarket(device='cpu')