import pandas as pd
import numpy as np
from ..utils.logging_utils import log_execution, handle_exceptions, logger
import logging

import math

class PrepareData:
    def __init__(self, data_path: str, join_keys: list, from_disk: bool, df: pd.DataFrame = None):
        if from_disk:
            # Check the first row to see if the data has the corresponding columns
            header_df = pd.read_csv(data_path, nrows=0)
            header_cols = set(header_df.columns)
            self.join_key_valid = False
            for join_key in join_keys:
                if set(join_key).issubset(header_cols):
                    self.join_key_valid = True
                    break
            if not self.join_key_valid:
                raise ValueError("Join keys column(s) provided are not present in the data.")
            self.data = pd.read_csv(data_path)
            logger.info("Data loaded from disk.")
        else:
            self.data = df
            logger.info("Data initialized from DataFrame.")
        
        self.join_keys = []
        self.join_keys_in_string = []
        self.has_key = self._check_join_keys(join_keys=join_keys)

        self._construct_join_keys()
    
    @log_execution(logging.DEBUG)
    def get_data(self):
        return self.data
    
    @log_execution(logging.DEBUG)
    def set_data(self, data: pd.DataFrame):
        self.data = data
    
    @log_execution(logging.DEBUG)
    def see_data(self):
        print(self.data.head())

    @log_execution(logging.INFO)
    def get_features(self):
        return self.data.columns
    
    @log_execution(logging.INFO)
    def see_features(self):
        print(self.data.columns)
    
    @log_execution(logging.INFO)
    def get_join_keys(self):
        return self.join_keys_in_string
    
    @log_execution(logging.DEBUG)
    def cut_data_by_features(self, features: list):
        return self.data[features]
    
    @log_execution(logging.DEBUG)
    def cut_data_by_join_keys(self, join_keys: list):
        return self.data[join_keys]
    
    @staticmethod
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _data_cleaning(df, join_keys_in_string):
        """Static method to perform data cleaning; calls the helper get_num_cols."""
        # Call the get_num_cols helper function
        return PrepareData.get_num_cols(df, set(join_keys_in_string))
    
    @staticmethod
    @handle_exceptions
    def get_num_cols(df, join_keys):
        """Helper static method to identify and clean numerical columns in DataFrame."""
        
        def is_numeric(val):
            # Check if value is NaN or not numeric
            if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
                return False
            return isinstance(val, (int, float, complex, np.integer, np.floating)) and not isinstance(val, bool)

        # Convert columns to numeric where applicable
        for col in df.columns:
            if col not in join_keys:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        num_cols = [col for col in df.columns if is_numeric(df[col].iloc[0])]
        display_cols = [col for col in df.columns]
        
        # Remove columns with > 40% missing values
        for col in df.columns:
            nan_fraction = df[col].apply(lambda x: x == '' or pd.isna(x)).mean()
            if nan_fraction > 0.4:
                display_cols.remove(col)

        # Fill NaNs with 0 for now
        df.fillna(0, inplace=True)

        for col in num_cols[:]:
            has_string = df[col].apply(lambda x: isinstance(x, str)).any()
            if has_string:
                non_numeric_fraction = df[col].apply(lambda x: not is_numeric(x)).mean()
                if non_numeric_fraction > 0.5:
                    num_cols.remove(col)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].mean(), inplace=True)

        # Check if join keys are in DataFrame columns
        if join_keys.difference(set(df.columns)): 
            return [], None
        else:
            for ele in join_keys.difference(set(display_cols)):
                display_cols = [ele] + display_cols
        return num_cols
    
    """
    join_keys is expected to be a list of list of strings (e.g. [["country", "year"], ["country"], ["year"]])
    for each list, we need to check if it is a subset of the column names.
    If not, we do not add the join_keys
    If yes, we add the join_keys
    """
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _check_join_keys(self, join_keys: list):
        """Private method to validate join keys against DataFrame columns."""
        for join_key in join_keys:
            if set(join_key).issubset(set(self.data.columns)):
                self.join_keys.append(join_key)

        if self.join_keys == []:
            return False
        return True
    
    """
    For all combination of join keys, we would make a new column that contains both of the values
    Like if we have join_keys = [["country", "year"], ["country"], ["year"]], then we would eventually have
    ["country_year", "country", "year"] as the join keys. All the other columns would be the same.
    """
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _construct_join_keys(self):

        for join_key in self.join_keys:
            if len(join_key) > 1:
                self.data["_".join(join_key)] = self.data[join_key].astype(str).apply(lambda x: "_".join(x), axis=1)
                self.join_keys_in_string.append("_".join(join_key))
            else:
                self.join_keys_in_string.append(join_key[0])
        return self.data