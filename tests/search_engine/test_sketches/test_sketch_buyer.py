import pytest
import pandas as pd
import torch
from search_engine.sketches.base import SketchBase
from search_engine.sketches.sketch_buyer import BuyerSketch

@pytest.fixture
def setup_buyer():
    join_key_domain = {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    sketch_base = SketchBase(join_key_domain=join_key_domain, device='cpu', is_buyer=True)
    buyer_df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    join_keys = ["country", "year"]
    target_feature = "value"
    return BuyerSketch(buyer_df=buyer_df, join_keys=join_keys, join_key_domains=join_key_domain, sketch_base=sketch_base, target_feature=target_feature, device='cpu')

def test_init_buyer(setup_buyer):
    buyer = setup_buyer
    assert buyer.device == 'cpu'
    assert buyer.join_keys == ["country", "year"]
    assert buyer.target_feature == "value"
    assert isinstance(buyer.sketch_base, SketchBase)

def test_register_this_buyer(setup_buyer):
    buyer = setup_buyer
    batch_id, offset = buyer.register_this_buyer()
    assert batch_id == 0
    assert offset == 0

def test_get_base(setup_buyer):
    buyer = setup_buyer
    base = buyer.get_base()
    assert isinstance(base, SketchBase)

def test_get_sketches(setup_buyer):
    buyer = setup_buyer
    buyer.register_this_buyer()
    sketches = buyer.get_sketches()
    assert sketches is not None

def test_get_target_feature(setup_buyer):
    buyer = setup_buyer
    target_feature = buyer.get_target_feature()
    assert target_feature["name"] == "value"
