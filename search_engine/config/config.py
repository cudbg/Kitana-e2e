from dataclasses import dataclass
import yaml
import os
from pathlib import Path
import logging

@dataclass
class SearchConfig:
    fit_by_residual: bool = True
    device: str = 'cpu'
    batch_size: str = 'auto'
    sketch_dir: str = 'sketches/'
    iterations: int = 10

@dataclass
class DataConfig:
    directory_path: str = 'data/test_data_small/seller'
    buyer_csv: str = 'data/test_data_small/buyer/buyer.csv'
    join_keys: list = None
    target_feature: str = 'b1'
    one_target_feature: bool = False
    need_to_clean_data: bool = True
    entity_linking_percentage: list = None

    def __post_init__(self):
        if self.join_keys is None:
            self.join_keys = [['m'], ['n']]
        if self.entity_linking_percentage is None:
            self.entity_linking_percentage = [100]

@dataclass
class ExperimentConfig:
    plot_results: bool = True
    save_results: bool = True
    results_dir: str = 'results/'

@dataclass
class LoggingConfig:
    level: str = 'DEBUG'
    file: str = 'logs/experiment.log'
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.file),
                logging.StreamHandler()
            ]
        )

@dataclass 
class Config:
    search: SearchConfig = SearchConfig()
    data: DataConfig = DataConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    logging: LoggingConfig = LoggingConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """load configuration from yaml file"""
        config = cls()
        
        if not os.path.exists(yaml_path):
            config.logging.setup_logging()
            return config
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
            
        # Create a new config object with the values from the yaml file
        new_config = cls(
            search=SearchConfig(**config_dict.get('search', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
        
        # Setup logging
        new_config.logging.setup_logging()
        return new_config

_config = None

def get_config(config_path: str = None) -> Config:
    """Get the configuration object"""
    global _config
    if _config is None:
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        else:
            if not os.path.isabs(config_path):
                config_path = Path(__file__).parent / config_path
        _config = Config.from_yaml(str(config_path))
    return _config