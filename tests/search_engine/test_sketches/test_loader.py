import pytest
import torch
from search_engine.sketches.loader import SketchLoader
import os

@pytest.fixture
def setup_loader():
    batch_size = 2
    device = 'cpu'
    disk_dir = 'test_sketches/'
    is_buyer = False
    return SketchLoader(batch_size=batch_size, device=device, disk_dir=disk_dir, is_buyer=is_buyer)

def test_init_loader(setup_loader):
    loader = setup_loader
    assert loader.batch_size == 2
    assert loader.device == 'cpu'
    assert loader.disk_dir == 'test_sketches/'
    assert not loader.is_buyer

def test_load_sketches_to_disk(setup_loader, mocker):
    loader = setup_loader
    seller_1 = torch.randn(3, 4)
    seller_x = torch.randn(3, 4)
    seller_x_x = torch.randn(3, 4)
    feature_index_map = {}
    seller_id = 1
    mocker.patch('torch.save')
    mocker.patch('os.makedirs')
    loader.load_sketches(seller_1, seller_x, seller_x_x, feature_index_map, seller_id, to_disk=True)
    assert len(loader.sketch_1_batch) > 0 or os.path.exists(loader.disk_dir)
    assert len(loader.sketch_x_batch) > 0 or os.path.exists(loader.disk_dir)
    assert len(loader.sketch_x_x_batch) > 0 or os.path.exists(loader.disk_dir)

def test_load_sketches_no_disk(setup_loader):
    loader = setup_loader
    seller_1 = torch.randn(3, 4)
    seller_x = torch.randn(3, 4)
    seller_x_x = torch.randn(3, 4)
    feature_index_map = {}
    seller_id = 1
    loader.load_sketches(seller_1, seller_x, seller_x_x, feature_index_map, seller_id, to_disk=False)
    assert len(loader.sketch_1_batch) > 0
    assert len(loader.sketch_x_batch) > 0
    assert len(loader.sketch_x_x_batch) > 0

def test_get_sketches_from_disk(setup_loader, mocker):
    loader = setup_loader
    seller_1 = torch.randn(3, 4)
    seller_x = torch.randn(3, 4)
    seller_x_x = torch.randn(3, 4)
    feature_index_map = {}
    seller_id = 1
    loader.load_sketches(seller_1, seller_x, seller_x_x, feature_index_map, seller_id, to_disk=True)
    
    mocker.patch('torch.load', side_effect=lambda path: torch.randn(3, 2))
    sketch_1_batch, sketch_x_batch, sketch_x_x_batch, sketch_x_y_batch = loader.get_sketches(0, from_disk=True)
    assert sketch_1_batch is not None
    assert sketch_x_batch is not None
    assert sketch_x_x_batch is not None

def test_get_sketches_no_disk(setup_loader):
    loader = setup_loader
    seller_1 = torch.randn(3, 4)
    seller_x = torch.randn(3, 4)
    seller_x_x = torch.randn(3, 4)
    feature_index_map = {}
    seller_id = 1
    loader.load_sketches(seller_1, seller_x, seller_x_x, feature_index_map, seller_id, to_disk=False)
    sketch_1_batch, sketch_x_batch, sketch_x_x_batch, sketch_x_y_batch = loader.get_sketches(0, from_disk=False)
    assert sketch_1_batch is not None
    assert sketch_x_batch is not None
    assert sketch_x_x_batch is not None

def test_get_num_batches(setup_loader):
    loader = setup_loader
    seller_1 = torch.randn(3, 4)
    seller_x = torch.randn(3, 4)
    seller_x_x = torch.randn(3, 4)
    feature_index_map = {}
    seller_id = 1
    loader.load_sketches(seller_1=seller_1,
                            seller_x=seller_x,
                            seller_x_x=seller_x_x,
                            feature_index_map=feature_index_map,
                            seller_id=seller_id,
                            to_disk=False)
    assert loader.get_num_batches() > 0
