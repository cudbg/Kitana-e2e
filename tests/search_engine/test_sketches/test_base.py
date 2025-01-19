import pytest
import pandas as pd
import torch
from search_engine.sketches.base import SketchBase
from search_engine.sketches.loader import SketchLoader
from search_engine.config.config import get_config

@pytest.fixture
def setup_base():
    config = get_config("test_config.yaml")
    join_key_domain = {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    return SketchBase(join_key_domain=join_key_domain, device=config.search.device, is_buyer=False)

def test_init_base(setup_base):
    base = setup_base
    config = get_config("test_config.yaml")
    assert base.device == config.search.device
    assert base.join_key_domain == {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    assert isinstance(base.sketch_loader, SketchLoader)

def test_register_df(setup_base):
    base = setup_base
    df_id = 1
    feature_num = 3
    seller_1 = torch.randn(3, 3)
    seller_x = torch.randn(3, 3)
    seller_x_x = torch.randn(3, 3)
    result = base._register_df(df_id, feature_num, seller_1, seller_x, seller_x_x, to_disk=False)
    assert result["df_id"] == df_id
    assert result["batch_id"] == 0
    assert result["offset"] == 0

def test_calibrate(setup_base):
    base = setup_base
    df_id = 1
    df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022],
        "value": [1, 2, 3]
    })
    num_features = 1
    key_domains = {
        "country": ["A", "B", "C"],
        "year": [2020, 2021, 2022]
    }
    join_keys = ["country", "year"]
    seller_x, seller_x_x, seller_1, seller_x_y = base._calibrate(df_id, df, num_features, key_domains, join_keys)
    assert seller_x is not None
    assert seller_x_x is not None
    assert seller_1 is not None

def test_get_df_by_feature_index(setup_base):
    base = setup_base
    df_id = 1
    feature_num = 3
    seller_1 = torch.randn(3, 3)
    seller_x = torch.randn(3, 3)
    seller_x_x = torch.randn(3, 3)
    base._register_df(df_id, feature_num, seller_1, seller_x, seller_x_x, to_disk=False)
    base.dfid_feature_mapping[df_id] = ["feature1", "feature2", "feature3"]
    df_id, feature_name = base.get_df_by_feature_index(0, 1)
    assert df_id == 1
    assert feature_name == "feature2"

def test_get_sketch_loader(setup_base):
    base = setup_base
    loader = base.get_sketch_loader()
    assert isinstance(loader, SketchLoader)
