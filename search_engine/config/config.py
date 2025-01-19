from dataclasses import dataclass
import yaml
import os
from pathlib import Path

@dataclass
class SearchConfig:
    """Search engine configuration"""
    fit_by_residual: bool = False
    device: str = 'cpu'
    batch_size: str = 'auto'
    sketch_dir: str = 'sketches/'

@dataclass 
class Config:
    """Global configuration settings"""
    search: SearchConfig = SearchConfig()
    log_level: str = 'INFO'
    data_dir: str = 'data/'
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file"""
        if not os.path.exists(yaml_path):
            return cls()
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        search_config = SearchConfig(**config_dict.get('search', {}))
        return cls(
            search=search_config,
            log_level=config_dict.get('log_level', 'INFO'),
            data_dir=config_dict.get('data_dir', 'data/')
        )

_config = None

def get_config(config_path: str = None) -> Config:
    """Get global configuration singleton"""
    global _config
    if _config is None:
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        _config = Config.from_yaml(str(config_path))
    return _config