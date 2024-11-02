from .base_data import PrepareData
from utils.logging_utils import log_execution, handle_exceptions
import pandas as pd
import logging

class PrepareBuyer(PrepareData):
    @log_execution(logging.INFO)
    def __init__(self, data_path: str, join_keys: list, target_feature: str, features: list = [], one_target_feature: bool = True, from_disk: bool = True, buyer_df: pd.DataFrame = None, need_to_clean_data: bool = True):
        """
        Initializes the PrepareBuyer class by loading data, performing cleaning, and setting up key domains.

        Parameters:
            data_path (str): Path to the dataset file.
            join_keys (list): List of keys used for merging data.
            target_feature (str): Main feature of interest.
            features (list): List of additional features to retain.
            one_target_feature (bool): If True, only target feature will be retained.
            from_disk (bool): Whether to load the data from disk.
            buyer_df (pd.DataFrame): DataFrame provided directly if from_disk is False.
            need_to_clean_data (bool): Flag to determine if data cleaning is required.
        """
        super().__init__(data_path, join_keys, from_disk=from_disk, df=buyer_df)
        self.from_disk = from_disk
        self.target_feature = target_feature

        if need_to_clean_data:
            num_cols = self._data_cleaning()
        else:
            num_cols = features

        # Ensure that the data includes only necessary columns
        self.data = self.data[list(set(self.join_keys_in_string).union(set(num_cols)).union({self.target_feature}))]

        self.one_target_feature = one_target_feature
        if self.one_target_feature:
            # Limit data to join keys and target feature if specified
            self.data = self.data[list(set(self.join_keys_in_string).union({self.target_feature}))]

        self.buyer_key_domain = {}
        self._calculate_buyer_key_domain()

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _calculate_buyer_key_domain(self):
        """
        Calculates the domains for each buyer key to facilitate further analysis or joins.
        """
        for join_key in self.join_keys_in_string:
            self.buyer_key_domain[join_key] = set(self.data[join_key])
        return self.buyer_key_domain
