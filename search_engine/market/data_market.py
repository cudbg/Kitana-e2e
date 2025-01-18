"""
This class would be used as a overall register for both seller and buyer sketches. It helps to prepare the 
buyer sketch base and seller sketch base for the search engine.
"""
import pandas as pd
import copy
import logging
from ..sketches.base import SketchBase
from ..sketches.sketch_buyer import BuyerSketch
from ..sketches.sketch_seller import SellerSketch
from ..statistics.statistics import linear_regression_residuals
from ..utils.logging_utils import log_execution, handle_exceptions

class DataMarket():
    """
    A class that serves as a central registry for both seller and buyer sketches.
    It manages the preparation and organization of sketch bases for the search engine.
    """
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, device='cpu'):
        # Storage for sketch bases
        self.seller_sketches = {}       # Format: join_key: [
                                       #           {id, join_key, join_key_domain, seller_sketch},
                                       #           {id, join_key, join_key_domain, seller_sketch}
                                       #         ]

        self.buyer_sketches = {}        # Format: join_key: {id, join_key, join_key_domain, buyer_sketch}
        self.buyer_dataset_for_residual = None
        
        # ID management for sellers and buyers
        self.seller_id = 0
        self.buyer_id = 0

        # Buyer-specific attributes
        self.buyer_target_feature = ""
        self.buyer_join_keys = []

        # Storage for dataset metadata
        self.seller_id_to_df_and_name = []
        self.buyer_id_to_df_and_name = []

        self.augplan_acc = []
        self.device = device
        
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def register_seller(self, seller_df: pd.DataFrame, seller_name: str, join_keys: list, join_key_domains: dict):
        """
        Registers a seller dataframe to the data market.
        Creates SellerSketch objects for each join key and updates internal registries.

        Args:
            seller_df: The seller's dataframe
            seller_name: Name identifier for the seller
            join_keys: List of join keys
            join_key_domains: Dictionary of domains for each join key

        Returns:
            int: The assigned seller ID
        """
        logging.info(f"Registering seller {seller_name} with {len(join_keys)} join keys")
        # To avoid the case where the seller_df are containing some features that have the same name as the registered features, 
        # we need to add a prefix for all the features except the join keys
        prefix = seller_name + "_"
        seller_df.columns = [prefix + col if col not in join_keys else col for col in seller_df.columns]
        for join_key in join_keys:
            logging.debug(f"Processing join key: {join_key}")
            if join_key in self.seller_sketches:
                seller_sketch_base = self.seller_sketches[join_key]["sketch_base"]
            else:
                seller_sketch_base = SketchBase(join_key_domain=join_key_domains, device=self.device)
                # Create a new list for the new join key
                self.seller_sketches[join_key] = {}
                self.seller_sketches[join_key]["sketch_base"] = seller_sketch_base
            seller_df_with_the_key = seller_df[list(seller_df.columns.difference(join_keys)) + [join_key]]
            # Create a SellerSketch object
            seller_sketch = SellerSketch(
                seller_df_with_the_key, 
                [join_key], 
                join_key_domains, 
                seller_sketch_base,
                self.seller_id, 
                self.device
            )
            # Register the seller and store the seller sketch object with related information
            seller_sketch_info = {}
            seller_sketch_info["id"] = self.seller_id
            seller_sketch_info["name"] = seller_name
            seller_sketch_info["join_key"] = join_key
            seller_sketch_info["join_key_domain"] = join_key_domains
            seller_sketch_info["seller_sketch"] = seller_sketch
            self.seller_sketches[join_key][self.seller_id] = seller_sketch_info

            
            batch_id, offset = seller_sketch.register_this_seller()

        self.seller_id_to_df_and_name.append(
            {"name": seller_name,
             "dataframe": seller_df}
        )
        # Update the seller_id
        self.seller_id += 1

        logging.info(f"Successfully registered seller {seller_name} with ID {self.seller_id}")
        return self.seller_id - 1
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def register_buyer(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict, target_feature: str, fit_by_residual=False):
        """
        Registers a buyer dataframe to the data market.
        Creates BuyerSketch objects for each join key and updates internal registries.

        Args:
            buyer_df: The buyer's dataframe
            join_keys: List of join keys
            join_key_domains: Dictionary of domains for each join key
            target_feature: Target feature for prediction
            fit_by_residual: Whether to fit by residual

        Returns:
            int: The assigned buyer ID
        """
        logging.info(f"Registering buyer with target feature {target_feature}")
        if fit_by_residual:
            self.buyer_dataset_for_residual = copy.deepcopy(buyer_df)
        self.buyer_dataset = copy.deepcopy(buyer_df)
        self.buyer_join_keys = join_keys
        self.buyer_target_feature = target_feature
        X = list(self.buyer_dataset.columns.difference([target_feature] + join_keys))
        # Calculate the residuals from linear regression
        res, r2 = linear_regression_residuals(self.buyer_dataset, X_columns=X, Y_column=target_feature, adjusted=False)
        self.augplan_acc.append(r2)
        if fit_by_residual:
            self.buyer_dataset = res[join_keys + ["residuals"]]
        else:
            self.buyer_dataset = self.buyer_dataset.drop(columns=["residuals"], errors="ignore")

        for join_key in join_keys:
            logging.debug(f"Processing buyer join key: {join_key}")
            if join_key in self.buyer_sketches:
                buyer_sketch_base = self.buyer_sketches[join_key]["buyer_sketch"].get_base()
            else:
                buyer_sketch_base = SketchBase(join_key_domain=join_key_domains, device=self.device, is_buyer=True)
            buyer_df_with_the_key = self.buyer_dataset[list(self.buyer_dataset.columns.difference(join_keys)) + [join_key]]
            # Create a BuyerSketch object
            buyer_sketch = BuyerSketch(
                buyer_df_with_the_key, 
                [join_key], 
                join_key_domains, 
                buyer_sketch_base, 
                target_feature, 
                self.device,
                fit_by_residual
            )
            # Register the buyer and store the buyer sketch object with related information
            buyer_sketch_info = {}
            buyer_sketch_info["id"] = self.buyer_id
            buyer_sketch_info["join_key"] = join_key
            buyer_sketch_info["join_key_domain"] = join_key_domains
            buyer_sketch_info["buyer_sketch"] = buyer_sketch
            self.buyer_sketches[join_key] = buyer_sketch_info

            batch_id, offset = buyer_sketch.register_this_buyer(fit_by_residual=fit_by_residual) # Currently, batch_id and offset are not used

        self.buyer_id_to_df_and_name.append(
            {"name": target_feature,
             "dataframe": self.buyer_dataset}
        )
        # Update the buyer_id
        self.buyer_id += 1

        logging.info(f"Successfully registered buyer with ID {self.buyer_id}")
        return self.buyer_id - 1
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def get_buyer_sketch(self, buyer_id):
        """Returns the buyer sketch object for a given buyer ID"""
        return self.buyer_sketches[buyer_id]["buyer_sketch"]
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def get_seller_sketch_by_keys(self, join_key, seller_id):
        """Returns the seller sketch object for given join key and seller ID"""
        return self.seller_sketches[join_key][seller_id]["seller_sketch"]
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def get_seller_sketch_base_by_keys(self, join_key):
        """Returns the seller sketch base for a given join key"""
        return self.seller_sketches[join_key]["sketch_base"]
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def get_buyer_sketch_by_keys(self, join_key):
        """Returns the buyer sketch object for a given join key"""
        return self.buyer_sketches[join_key]["buyer_sketch"]
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def set_buyer_id(self, buyer_id):
        """Sets the buyer ID"""
        logging.debug(f"Setting buyer ID to {buyer_id}")
        self.buyer_id = buyer_id

    @handle_exceptions
    @log_execution(logging.INFO)
    def reset_buyer_sketches(self):
        """Resets the buyer sketches dictionary"""
        logging.info("Resetting buyer sketches")
        self.buyer_sketches = {}

    @handle_exceptions
    @log_execution(logging.INFO)
    def reset_buyer_id_to_df_and_name(self):
        """Resets the buyer ID to dataframe and name mapping"""
        logging.info("Resetting buyer ID to dataframe and name mapping")
        self.buyer_id_to_df_and_name = []