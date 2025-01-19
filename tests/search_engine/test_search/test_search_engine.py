import pytest
from search_engine.search.search_engine import SearchEngine
from search_engine.config.config import get_config
                                                                            
def test_search_engine_init(setup_market):
    engine = SearchEngine(setup_market)
    config = get_config("test_config.yaml")
    assert engine.data_market is not None
    assert engine.buyer_target == "target"
    assert len(engine.augplan) == 0
    assert len(engine.augplan_acc) == 0
    assert engine.fit_by_residual == config.search.fit_by_residual
    assert engine.device == config.search.device

def test_search_one_iteration(setup_market):
    engine = SearchEngine(setup_market)
    join_key, feature_ind, batch_id = engine.search_one_iteration()
    
    assert join_key in setup_market.buyer_sketches
    assert isinstance(feature_ind, int)
    assert isinstance(batch_id, int)
    
    # Get the feature name
    seller_id, feature_name = setup_market.get_seller_sketch_base_by_keys(join_key).get_df_by_feature_index(batch_id, feature_ind)
    assert feature_name is not None

def test_full_search(setup_market):
    engine = SearchEngine(setup_market)
    augplan, augplan_acc, final_dataset = engine.start(iter=2)
    
    assert len(augplan) > 0
    assert len(augplan_acc) > 0
    assert final_dataset is not None
    
    # Check augplan format
    for plan in augplan:
        seller_id, iter_num, seller_name, feature = plan
        assert isinstance(seller_id, int)
        assert isinstance(iter_num, int)
        assert isinstance(seller_name, str)
        assert isinstance(feature, str)

def test_search_with_residual(setup_market_with_residual):
    engine = SearchEngine(setup_market_with_residual)
    engine.fit_by_residual = True
    augplan, augplan_acc, final_dataset = engine.start(iter=2)
    
    assert len(augplan) > 0
    assert len(augplan_acc) > 0
    assert final_dataset is not None

def test_unusable_features_handling(setup_market):
    engine = SearchEngine(setup_market)
    
    # Force some features to be unusable
    engine.unusable_features = {0: [0, 1]} 
    
    join_key, feature_ind, batch_id = engine.search_one_iteration()
    
    # Check that unusable features are not selected
    if batch_id == 0:
        assert feature_ind not in [0, 1]

def test_update_residual(setup_market):
    engine = SearchEngine(setup_market)
    
    # Get initial state
    initial_columns = set(setup_market.buyer_dataset.columns)
    
    # Perform one iteration
    join_key, feature_ind, batch_id = engine.search_one_iteration()
    seller_id, feature = setup_market.get_seller_sketch_base_by_keys(join_key).get_df_by_feature_index(batch_id, feature_ind)
    
    # Update residual
    engine._update_residual(join_key, seller_id, feature)
    
    # Check that new feature is added
    updated_columns = set(setup_market.buyer_dataset.columns)
    assert len(updated_columns) > len(initial_columns)
    assert feature in updated_columns