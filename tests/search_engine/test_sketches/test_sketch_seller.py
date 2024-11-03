import pytest
import pandas as pd
import torch
from search_engine.sketches.base import SketchBase
from search_engine.sketches.sketch_seller import SellerSketch

@pytest.fixture
def setup_seller():
    join_key_domain = {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    sketch_base = SketchBase(join_key_domain=join_key_domain, device='cpu', is_buyer=False)
    seller_df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    join_keys = ["country", "year"]
    return SellerSketch(seller_df=seller_df, join_keys=join_keys, join_key_domains=join_key_domain, sketch_base=sketch_base, df_id=1, device='cpu')

def test_init_seller(setup_seller):
    seller = setup_seller
    assert seller.device == 'cpu'
    assert seller.join_keys == ["country", "year"]
    assert isinstance(seller.sketch_base, SketchBase)

def test_register_this_seller(setup_seller):
    seller = setup_seller
    batch_id, offset = seller.register_this_seller()
    assert batch_id == 0
    assert offset == 0

def test_get_base(setup_seller):
    seller = setup_seller
    base = seller.get_base()
    assert isinstance(base, SketchBase)

def test_get_sketches(setup_seller):
    seller = setup_seller
    seller.register_this_seller()
    sketches = seller.get_sketches()
    assert sketches is not None
