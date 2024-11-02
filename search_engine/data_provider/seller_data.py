# Import necessary modules
from .base_data import PrepareData
from ..utils.logging_utils import log_execution, handle_exceptions
import pandas as pd
import logging

class PrepareSeller(PrepareData):
    @log_execution(logging.INFO)
    def __init__(self, data_path: str, join_keys: list, features: list = [], from_disk: bool = True, seller_df: pd.DataFrame = None, need_to_clean_data: bool = True):
        """
        Initializes the PrepareSeller class by loading data, performing cleaning, and preparing domains.

        Parameters:
            data_path (str): Path to the dataset file.
            join_keys (list): List of keys used for merging data.
            features (list): List of additional features to retain.
            from_disk (bool): Whether to load the data from disk.
            seller_df (pd.DataFrame): DataFrame provided directly if from_disk is False.
            need_to_clean_data (bool): Flag to determine if data cleaning is required.
        """
        super().__init__(data_path, join_keys, from_disk=from_disk, df=seller_df)
        self.key_domain_recorded = False
        self.join_key_domains = {} # {join_key: {set of values}}

        if need_to_clean_data:
            num_cols = self._data_cleaning(self.data, self.join_keys_in_string)
        else:
            num_cols = features

        # Ensure that the data includes only necessary columns
        self.data = self.data[list(set(self.join_keys_in_string).union(set(num_cols)))]

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_record_status(self):
        """
        Returns the recording status of the join key domains.
        """
        return self.key_domain_recorded
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def set_join_key_domains(self, join_key: str, domain: set):
        """
        Sets the domain for a specified join key.
        """
        self.join_key_domains[join_key] = domain.union(self.data[join_key])

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_join_key_domains(self, join_key: str):
        """
        Gets the domain of a specified join key.
        """
        return self.join_key_domains[join_key]

    @handle_exceptions
    @log_execution(logging.INFO)
    def record_join_key_domains(self):
        """
        Records the domains for each join key in the data.
        """
        for join_key in self.join_keys_in_string:
            self.set_join_key_domains(join_key, set(self.data[join_key]))
        self.key_domain_recorded = True
        return self.join_key_domains
