import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from ..data_provider import PrepareBuyerSellers, PrepareBuyer, PrepareSeller
from ..market import DataMarket
from ..search import SearchEngine
from ..utils.logging_utils import log_execution, handle_exceptions
from ..utils.plot_utils import plot_whiskers
from ..entity_linking.el_test import DBpediaLinker
import logging

class ScaledExperiment:
    """Handles scaled data market experiments with multiple sellers and buyers."""
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, config):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Configuration object with experiment settings
        """
        self.config = config
        logging.info(f"Experiment configuration: {config}")
        self.prepare_data = PrepareBuyerSellers()
        self.data_market = None
        self.search_engine = None
        self.results = {
            'augplan': None,
            'accuracy': None,
            'time_taken': None
        }

    @handle_exceptions
    @log_execution(logging.INFO)
    def load_buyer(self):
        """Load and prepare buyer data."""
        buyer = PrepareBuyer(
            data_path=self.config.data.buyer_csv,
            join_keys=self.config.data.join_keys,
            target_feature=self.config.data.target_feature,
            one_target_feature=self.config.data.one_target_feature,
            need_to_clean_data=self.config.data.need_to_clean_data
        )
        self.prepare_data.add_buyer(buyer)
        return buyer

    @handle_exceptions
    @log_execution(logging.INFO)
    def load_sellers(self):
        """Load and prepare seller data from directory."""
        seller_paths = [
            os.path.join(self.config.data.directory_path, f) 
            for f in os.listdir(self.config.data.directory_path) 
            if f.endswith(".csv")
        ]
        
        for seller_path in tqdm(seller_paths, desc="Loading sellers"):
            try:
                seller_name = os.path.basename(seller_path).replace('.csv', '')
                seller = PrepareSeller(
                    data_path=seller_path,
                    join_keys=self.config.data.join_keys,
                    need_to_clean_data=self.config.data.need_to_clean_data
                )
                self.prepare_data.add_seller(seller_name=seller_name, seller=seller)
            except Exception as e:
                logging.error(f"Error processing {seller_path}: {str(e)}")

    @handle_exceptions
    @log_execution(logging.INFO)
    def setup_market(self):
        """Initialize and setup the data market."""

        valid_join_key, join_key_changed = self.prepare_data.filter_join_keys(self.config.data.join_keys)
        
        if len(valid_join_key) == 0:
            raise ValueError("No valid join keys found in sellers")
            
        buyer_join_keys = valid_join_key if join_key_changed else list(self.prepare_data.get_buyer_join_keys())
        self.data_market = DataMarket()
        self.data_market.register_buyer(
            buyer_df=self.prepare_data.get_buyer_data(),
            join_keys=buyer_join_keys,
            target_feature=self.config.data.target_feature,
            join_key_domains=self.prepare_data.get_domain()
        )
        logging.info(f"Buyer registered, data: {self.prepare_data.get_buyer_data().shape}, columns: {self.prepare_data.get_buyer_data().columns}")

        seller_data = self.prepare_data.get_seller_data()
        seller_names = self.prepare_data.get_sellers().get_seller_names()
        
        for seller_name in tqdm(seller_names, desc="Registering sellers"):
            seller = seller_data[seller_name]
            join_keys = list(seller.join_key_domains.keys())
            self.data_market.register_seller(
                seller_df=seller.get_data(),
                seller_name=seller_name,
                join_keys=join_keys,
                join_key_domains=self.prepare_data.get_domain()
            )

        logging.info(f"Data market setup complete with {len(seller_names)} sellers")
        logging.debug(f"First seller, data: {seller_data[seller_names[0]].get_data().shape}, columns: {seller_data[seller_names[0]].get_data().columns}")

    @handle_exceptions
    @log_execution(logging.INFO)
    def run_search(self):
        """Execute the search process."""
        self.search_engine = SearchEngine(
            data_market=self.data_market
        )
        
        start_time = time.time()
        augplan, accuracies, res_dataset = self.search_engine.start(iter=self.config.search.iterations)
        end_time = time.time()
        logging.info(f"The accuracy: {self.data_market.augplan_acc}")
        self.results = {
            'augplan': augplan,
            'accuracy': accuracies,
            'time_taken': end_time - start_time
        }

    @handle_exceptions
    @log_execution(logging.INFO)
    def plot_results(self):
        """Plot experiment results if enabled."""
        if not self.config.experiment.plot_results:
            return
        
        accuracy = self.results['accuracy']

        df = pd.DataFrame({
            'Epoch': np.arange(1, len(accuracy) + 1),
            'Accuracy': accuracy
        })

        # Assuming a fixed variability for illustrative purposes
        lower_bound = 0.01
        upper_bound = 0.01
        plot_whiskers(df, 'Epoch', ['Accuracy'], ['Model Accuracy'], ['blue'], ['-'], figsize=(12, 8), resultname=f'{self.config.experiment.results_dir}/accuracy_plot.png')


    @handle_exceptions
    @log_execution(logging.INFO)
    def run(self):
        """Execute the complete experiment workflow."""
        logging.info("Starting scaled experiment")
        
        self.load_buyer()
        self.load_sellers()
        self.setup_market()
        self.run_search()
        
        logging.info(f"Search completed in {self.results['time_taken']:.2f} seconds")
        logging.info(f"Final accuracy: {self.results['accuracy']}")
        
        self.plot_results()
        return self.results
